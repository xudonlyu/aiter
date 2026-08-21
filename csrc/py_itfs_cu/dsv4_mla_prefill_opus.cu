// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host-side loader for the prebuilt DSv4 MLA paged sparse-prefill code object.
// No device code here: the kernel ships as hsa/gfx950/dsv4_mla_prefill_opus/
// dsv4_mla_prefill_opus.co and this translation unit is the only thing aiter
// JIT-compiles for the op.
//
// Because the kernel header is deliberately not vendored next to the code
// object, the argument block below is a hand-maintained mirror.  The
// static_asserts are what keeps it honest: if the kernel's struct moves, the
// build breaks here instead of the kernel reading garbage at run time.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
// Last: it uses aiter_detail::g_aiter_can_throw, which the headers above define.
#include "aiter_ctypes_error.h"

#include <cstddef>

namespace {

constexpr const char* CO_PATH = "/dsv4_mla_prefill_opus/dsv4_mla_prefill_opus.co";
constexpr const char* SYM_PREFILL =
    "_Z39pa_prefill_16mx8_32nx1_fp8_paged_kernelI31pa_16mx8_32nx1_fp8_paged_"
    "traitsILi16ELi32ELi8EDB8_DF16bDF16bEEv18pa_fp8_paged_kargs";

// Geometry the code object was compiled with.  These are contract, not taste:
// the traits are baked into the object and the launcher cannot renegotiate them.
constexpr int D_NOPE      = 448;   // real NoPE width; Q arrives this wide, bf16
constexpr int D_ROPE      = 64;
constexpr int D_HEAD      = 512;
constexpr int ROW_BYTES   = 576;   // 448 NoPE fp8 + 64 RoPE bf16
constexpr int SCALE_SLOT  = 8;     // 7 per-64 UE8M0 + 1 pad, per token
constexpr int Q_TILE      = 16;
constexpr int T_M         = 8;
constexpr int BLOCK_SIZE  = 512;
constexpr int HEADS_PER_BLOCK = Q_TILE * T_M;   // 128

// Mirror of pa_fp8_paged_kargs.  Field order and padding must match exactly.
struct PaFp8PagedKargs
{
    const void* q_nope_ptr;
    const void* q_rope_ptr;
    const void* unified_kv_nope_ptr;
    const void* unified_kv_rope_ptr;
    const void* kv_nope_ptr;
    const void* kv_rope_ptr;
    const void* attn_sink_ptr;
    void*       out_ptr;
    const int*  kv_indptr_prefix;
    const int*  kv_indices_prefix;
    const int*  kv_indptr_extend;
    const int*  kv_indices_extend;
    const int*  kv_lens_prefix;
    const int*  kv_lens_extend;
    int   N;
    int   H;
    int   total_pages;
    int   total_tokens;
    int   stride_q_nope_n;
    int   stride_q_nope_h;
    int   stride_q_rope_n;
    int   stride_q_rope_h;
    int   stride_o_n;
    int   stride_o_h;
    int   stride_kv_nope_page;
    int   stride_kv_rope_page;
    int   kv_stride_q_prefix;
    int   kv_stride_q_extend;
    int   page_shift[2];
    int   rows_per_page[2];
    int   scale_off[2];
    float softmax_scale;
};

static_assert(sizeof(PaFp8PagedKargs) == 200, "kargs layout drifted from the .co");
static_assert(offsetof(PaFp8PagedKargs, out_ptr) == 56, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, kv_lens_prefix) == 96, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, N) == 112, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, stride_kv_nope_page) == 152, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, kv_stride_q_prefix) == 160, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, page_shift) == 168, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, rows_per_page) == 176, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, scale_off) == 184, "kargs layout drifted");
static_assert(offsetof(PaFp8PagedKargs, softmax_scale) == 192, "kargs layout drifted");

// Constructing the singleton is what registers the code object, and registering
// is illegal while a stream is capturing -- with cudagraphs the first call can
// land inside one.  The python wrapper forces it at import instead.
AiterAsmKernel& k_prefill()
{
    static AiterAsmKernel k(SYM_PREFILL, CO_PATH);
    return k;
}

inline int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

void paged_prefill_impl(aiter_tensor_t& q_nope,
                        aiter_tensor_t& q_rope,
                        aiter_tensor_t& unified_kv_nope,
                        aiter_tensor_t& unified_kv_rope,
                        aiter_tensor_t& kv_indices_prefix,
                        aiter_tensor_t& kv_indptr_prefix,
                        aiter_tensor_t& kv_nope,
                        aiter_tensor_t& kv_rope,
                        aiter_tensor_t& kv_indices_extend,
                        aiter_tensor_t& kv_indptr_extend,
                        aiter_tensor_t& attn_sink,
                        aiter_tensor_t& out,
                        float softmax_scale,
                        int page_shift_prefix,
                        int rows_per_page_prefix,
                        int scale_off_prefix,
                        int page_shift_extend,
                        int rows_per_page_extend,
                        int scale_off_extend,
                        aiter_tensor_t& kv_lens_prefix,
                        aiter_tensor_t& kv_lens_extend,
                        int kv_stride_q_prefix,
                        int kv_stride_q_extend,
                        hipStream_t stream)
{
    AITER_CHECK(q_nope.dim() == 3, "q_nope must be 3-D [N, H, 448], got ndim=", q_nope.dim());
    AITER_CHECK(q_rope.dim() == 3, "q_rope must be 3-D [N, H, 64], got ndim=", q_rope.dim());
    AITER_CHECK(unified_kv_nope.dim() == 2 && kv_nope.dim() == 2,
                "unified_kv_nope / kv_nope must be 2-D [rows, 512] views of the pool");
    AITER_CHECK(unified_kv_rope.dim() == 2 && kv_rope.dim() == 2,
                "unified_kv_rope / kv_rope must be 2-D [rows, 64] views of the pool");
    AITER_CHECK(out.dim() == 3, "out must be 3-D [N, H, 512], got ndim=", out.dim());
    AITER_CHECK(attn_sink.dim() == 1, "attn_sink must be 1-D [H]");

    // Q is bf16 while the KV streams are fp8 -- the two differ on purpose,
    // because the kernel packs Q itself in its prologue.
    AITER_CHECK(q_nope.dtype() == AITER_DTYPE_bf16,
                "q_nope must be bf16 [N, H, 448]; this kernel packs it");
    AITER_CHECK(unified_kv_nope.dtype() == AITER_DTYPE_fp8 && kv_nope.dtype() == AITER_DTYPE_fp8,
                "unified_kv_nope/kv_nope must be fp8");
    AITER_CHECK(q_rope.dtype() == AITER_DTYPE_bf16 && unified_kv_rope.dtype() == AITER_DTYPE_bf16 &&
                    kv_rope.dtype() == AITER_DTYPE_bf16,
                "q_rope/unified_kv_rope/kv_rope must be bf16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(attn_sink.dtype() == AITER_DTYPE_fp32, "attn_sink must be fp32");
    AITER_CHECK(kv_indptr_prefix.dtype() == AITER_DTYPE_i32 &&
                    kv_indices_prefix.dtype() == AITER_DTYPE_i32 &&
                    kv_indptr_extend.dtype() == AITER_DTYPE_i32 &&
                    kv_indices_extend.dtype() == AITER_DTYPE_i32,
                "kv_indptr / kv_indices must be int32");

    const int N = static_cast<int>(q_nope.size(0));
    const int H = static_cast<int>(q_nope.size(1));

    // The code object holds the T_M=8 pipeline only; H <= 32 is a different
    // pipeline with no paged variant.
    AITER_CHECK(H > 32,
                "the paged fp8 prefill is only compiled for H > 32 (T_M=8), got H=", H,
                "; route narrower head counts to pa_sparse_prefill_fp8_opus_fwd");

    AITER_CHECK(q_nope.size(2) == D_NOPE, "q_nope last dim must be 448");
    AITER_CHECK(q_rope.size(0) == N && q_rope.size(1) == H && q_rope.size(2) == D_ROPE,
                "q_rope shape must be [N, H, 64]");
    AITER_CHECK(unified_kv_nope.size(1) == D_HEAD && kv_nope.size(1) == D_HEAD,
                "the NoPE views must be 512 columns wide");
    AITER_CHECK(unified_kv_rope.size(1) == D_ROPE && kv_rope.size(1) == D_ROPE,
                "the RoPE views must be 64 columns wide");
    AITER_CHECK(out.size(0) == N && out.size(1) == H && out.size(2) == D_HEAD,
                "out shape must be [N, H, 512]");
    AITER_CHECK(attn_sink.size(0) == H, "attn_sink length must equal H");

    AITER_CHECK(q_nope.stride(2) == 1 && q_nope.stride(1) == D_NOPE,
                "q_nope must be contiguous with row stride 448");
    AITER_CHECK(q_rope.stride(2) == 1 && q_rope.stride(1) == D_ROPE,
                "q_rope must be contiguous with row stride 64");
    AITER_CHECK(out.stride(2) == 1, "out must be contiguous along the head dim");
    AITER_CHECK(unified_kv_nope.stride(1) == 1 && kv_nope.stride(1) == 1 &&
                    unified_kv_rope.stride(1) == 1 && kv_rope.stride(1) == 1,
                "the KV views must be unit-stride along their last dim");
    AITER_CHECK(kv_indices_prefix.is_contiguous() && kv_indptr_prefix.is_contiguous() &&
                    kv_indices_extend.is_contiguous() && kv_indptr_extend.is_contiguous() &&
                    attn_sink.is_contiguous(),
                "kv_indices/kv_indptr and attn_sink must be contiguous");

    if(N == 0)
        return;

    const int stride_kv_nope_page = static_cast<int>(unified_kv_nope.stride(0));
    const int stride_kv_rope_page = static_cast<int>(unified_kv_rope.stride(0));
    AITER_CHECK(stride_kv_nope_page == ROW_BYTES,
                "unified_kv_nope row stride must be ", ROW_BYTES, " fp8 elements, got ",
                stride_kv_nope_page);
    AITER_CHECK(stride_kv_rope_page * 2 == ROW_BYTES,
                "unified_kv_rope row stride must be ", ROW_BYTES / 2, " bf16 elements, got ",
                stride_kv_rope_page);
    AITER_CHECK(static_cast<int>(kv_nope.stride(0)) == stride_kv_nope_page &&
                    static_cast<int>(kv_rope.stride(0)) == stride_kv_rope_page,
                "the prefix and extend views must share the pool's row strides");

    PaFp8PagedKargs kargs{};

    const int shifts[2]   = {page_shift_prefix, page_shift_extend};
    const int rpp[2]      = {rows_per_page_prefix, rows_per_page_extend};
    const int soff[2]     = {scale_off_prefix, scale_off_extend};
    const int64_t rows[2] = {static_cast<int64_t>(unified_kv_nope.size(0)),
                             static_cast<int64_t>(kv_nope.size(0))};
    for(int sg = 0; sg < 2; ++sg)
    {
        // page_shift 0 is not the flat layout in disguise: a flat row carries
        // 14 per-32 exponents inline, this kernel always fetches 7 per-64 ones
        // from a page tail.  Accepting it would read RoPE bytes as exponents.
        AITER_CHECK(shifts[sg] >= 1 && shifts[sg] < 31,
                    "page_shift must be in [1, 31), got ", shifts[sg],
                    " (this kernel has no flat layout)");
        const int64_t page_size = static_cast<int64_t>(1) << shifts[sg];
        AITER_CHECK(rpp[sg] >= page_size,
                    "rows_per_page (", rpp[sg], ") must cover a page's ", page_size, " tokens");
        AITER_CHECK(soff[sg] >= page_size * ROW_BYTES,
                    "scale_off (", soff[sg], ") overlaps the page's token rows");
        // bytes_per_page = page_size*576 leaves the scale tail zero bytes long
        // and puts every scale write into the next page's fp8 data.
        AITER_CHECK(soff[sg] + page_size * SCALE_SLOT <=
                        static_cast<int64_t>(rpp[sg]) * ROW_BYTES,
                    "the scale tail does not fit in a page: scale_off=", soff[sg],
                    " page_size=", page_size,
                    " bytes_per_page=", static_cast<int64_t>(rpp[sg]) * ROW_BYTES);
        AITER_CHECK(rows[sg] % rpp[sg] == 0,
                    "the KV view (", rows[sg], " rows) is not a whole number of ", rpp[sg],
                    "-row pages");
        kargs.page_shift[sg]    = shifts[sg];
        kargs.rows_per_page[sg] = rpp[sg];
        kargs.scale_off[sg]     = soff[sg];
    }

    // Dense index form when lengths are supplied, CSR when they are not.
    const bool dense_p = kv_lens_prefix.numel() > 0;
    const bool dense_e = kv_lens_extend.numel() > 0;
    AITER_CHECK(dense_p == dense_e,
                "kv_lens_prefix and kv_lens_extend must both be given or both omitted");
    if(dense_p)
    {
        AITER_CHECK(kv_lens_prefix.dtype() == AITER_DTYPE_i32 &&
                        kv_lens_extend.dtype() == AITER_DTYPE_i32,
                    "kv_lens must be int32");
        AITER_CHECK(kv_lens_prefix.numel() == N && kv_lens_extend.numel() == N,
                    "kv_lens must have one entry per query token");
        AITER_CHECK(kv_lens_prefix.is_contiguous() && kv_lens_extend.is_contiguous(),
                    "kv_lens must be contiguous");
        AITER_CHECK(kv_stride_q_prefix > 0 && kv_stride_q_extend > 0,
                    "the dense form needs a positive row stride for the index array");
        AITER_CHECK(kv_indices_prefix.numel() >= static_cast<int64_t>(N) * kv_stride_q_prefix &&
                        kv_indices_extend.numel() >= static_cast<int64_t>(N) * kv_stride_q_extend,
                    "the dense index array is shorter than N * kv_stride_q");
    }
    else
    {
        AITER_CHECK(kv_indptr_prefix.size(0) == N + 1 && kv_indptr_extend.size(0) == N + 1,
                    "kv_indptr length must be N+1 unless the dense form is used");
    }

    kargs.q_nope_ptr          = q_nope.data_ptr();
    kargs.q_rope_ptr          = q_rope.data_ptr();
    kargs.unified_kv_nope_ptr = unified_kv_nope.data_ptr();
    kargs.unified_kv_rope_ptr = unified_kv_rope.data_ptr();
    kargs.kv_nope_ptr         = kv_nope.data_ptr();
    kargs.kv_rope_ptr         = kv_rope.data_ptr();
    kargs.attn_sink_ptr       = attn_sink.data_ptr();
    kargs.out_ptr             = out.data_ptr();
    kargs.kv_indptr_prefix    = reinterpret_cast<const int*>(kv_indptr_prefix.data_ptr());
    kargs.kv_indices_prefix   = reinterpret_cast<const int*>(kv_indices_prefix.data_ptr());
    kargs.kv_indptr_extend    = reinterpret_cast<const int*>(kv_indptr_extend.data_ptr());
    kargs.kv_indices_extend   = reinterpret_cast<const int*>(kv_indices_extend.data_ptr());
    kargs.kv_lens_prefix =
        dense_p ? reinterpret_cast<const int*>(kv_lens_prefix.data_ptr()) : nullptr;
    kargs.kv_lens_extend =
        dense_e ? reinterpret_cast<const int*>(kv_lens_extend.data_ptr()) : nullptr;
    kargs.N                   = N;
    kargs.H                   = H;
    kargs.total_pages         = static_cast<int>(rows[0]);
    kargs.total_tokens        = static_cast<int>(rows[1]);
    kargs.stride_q_nope_n     = static_cast<int>(q_nope.stride(0));
    kargs.stride_q_nope_h     = static_cast<int>(q_nope.stride(1));
    kargs.stride_q_rope_n     = static_cast<int>(q_rope.stride(0));
    kargs.stride_q_rope_h     = static_cast<int>(q_rope.stride(1));
    kargs.stride_o_n          = static_cast<int>(out.stride(0));
    kargs.stride_o_h          = static_cast<int>(out.stride(1));
    kargs.stride_kv_nope_page = stride_kv_nope_page;
    kargs.stride_kv_rope_page = stride_kv_rope_page;
    kargs.kv_stride_q_prefix  = kv_stride_q_prefix;
    kargs.kv_stride_q_extend  = kv_stride_q_extend;
    kargs.softmax_scale       = softmax_scale;

    HipDeviceGuard guard(q_nope.device_id);

    // LDS is static in this kernel (group_segment_fixed_size 135680), so the
    // dynamic shared-memory request stays 0.
    size_t sz = sizeof(kargs);
    k_prefill().launch_kernel({&kargs, &sz,
                               N, ceil_div_i(H, HEADS_PER_BLOCK), 1,
                               BLOCK_SIZE, 1, 1,
                               stream});
}

} // namespace

AITER_CTYPES_ERROR_DEF

AITER_C_ITFS int dsv4_mla_prefill_opus_warmup()
{
    return aiter_safe_call(g_aiter_last_error, [&] {
        k_prefill();
        return 0;
    });
}

AITER_C_ITFS int dsv4_mla_prefill_opus_fwd(aiter_tensor_t* q_nope,
                                           aiter_tensor_t* q_rope,
                                           aiter_tensor_t* unified_kv_nope,
                                           aiter_tensor_t* unified_kv_rope,
                                           aiter_tensor_t* kv_indices_prefix,
                                           aiter_tensor_t* kv_indptr_prefix,
                                           aiter_tensor_t* kv_nope,
                                           aiter_tensor_t* kv_rope,
                                           aiter_tensor_t* kv_indices_extend,
                                           aiter_tensor_t* kv_indptr_extend,
                                           aiter_tensor_t* attn_sink,
                                           aiter_tensor_t* out,
                                           float softmax_scale,
                                           int page_shift_prefix,
                                           int rows_per_page_prefix,
                                           int scale_off_prefix,
                                           int page_shift_extend,
                                           int rows_per_page_extend,
                                           int scale_off_extend,
                                           aiter_tensor_t* kv_lens_prefix,
                                           aiter_tensor_t* kv_lens_extend,
                                           int kv_stride_q_prefix,
                                           int kv_stride_q_extend,
                                           void* stream)
{
    return aiter_safe_call(g_aiter_last_error, [&] {
        paged_prefill_impl(*q_nope, *q_rope, *unified_kv_nope, *unified_kv_rope,
                           *kv_indices_prefix, *kv_indptr_prefix, *kv_nope, *kv_rope,
                           *kv_indices_extend, *kv_indptr_extend, *attn_sink, *out,
                           softmax_scale,
                           page_shift_prefix, rows_per_page_prefix, scale_off_prefix,
                           page_shift_extend, rows_per_page_extend, scale_off_extend,
                           *kv_lens_prefix, *kv_lens_extend,
                           kv_stride_q_prefix, kv_stride_q_extend,
                           static_cast<hipStream_t>(stream));
        return 0;
    });
}
