// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_hip_common.h"

struct __attribute__((packed)) TopKDecodeKernelArgs
{
    void* ptr_logits;
    void* ptr_seqLens;
    void* ptr_outIndices;
    int32_t stride0;
    int32_t stride1;
    int32_t next_n;
};

// Arg struct for the `_with_topk` decode kernel (TopKParams ABI). Must NOT be
// packed: the kernel reads `sizeof(TopKParams)` from the buffer, so packing the
// host side would shrink it to 44 bytes and corrupt the launch.
struct TopKDecodeKernelArgsWithTopk
{
    void*   ptr_buf;          // workspace
    void*   ptr_logits;
    void*   ptr_seqLens;
    void*   ptr_outIndices;
    int32_t stride0;
    int32_t stride1;
    int32_t next_n;
};
static_assert(sizeof(TopKDecodeKernelArgsWithTopk) == 48,
              "TopKDecodeKernelArgsWithTopk must match `TopKParams` (48 bytes)");

AITER_C_ITFS void top_k_per_row_decode_fast(
    aiter_tensor_t* logits,
    int64_t next_n,
    aiter_tensor_t* seqLens,
    aiter_tensor_t* indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    TopKDecodeKernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_logits     = logits->data_ptr();
    args.ptr_seqLens    = seqLens->data_ptr();
    args.ptr_outIndices = indices->data_ptr();
    args.stride0        = static_cast<int32_t>(stride0);
    args.stride1        = static_cast<int32_t>(stride1);
    args.next_n         = static_cast<int32_t>(next_n);

    static AiterAsmKernel impl_topk_decode(
        "_ZN5aiter10DecodeTopKL19topk_per_row_decodeILi1024ELb0ELi4EEEvPKfPKiPiiii",
        "/topk_per_row_decode/asm_top_k_per_row_decode.co");

    constexpr int kNumThreadsPerBlock = 1024;
    AITER_CHECK(numRows >> 31 == 0, "numRows too large: ", numRows);

    impl_topk_decode.launch_kernel({&args,
                                    &arg_size,
                                    static_cast<int>(numRows),
                                    1, 1,
                                    kNumThreadsPerBlock,
                                    1, 1,
                                    stream});
}

// ----------------------------------------------------------------------------
// `top_k_per_row_decode_fast_with_topk` — runtime-K variant of the asm-fast
// decode top-K kernel. Loads
// `topk_per_row_decode/asm_top_k_per_row_decode_with_topk.co` and dispatches by
// `topK` (512/1024/2048). Uses the TopKParams struct ABI. This is the path used
// on gfx950 (FULL cudagraph) where the base `top_k_per_row_decode_fast` .co is
// not shipped.
//
// The per-call scratch buffer (`topK * (sizeof(float) + sizeof(int32_t)) *
// numRows` bytes) is allocated by the caller (Python, via the torch caching
// allocator) and passed in as `workspace`, so this TU stays torch-free and the
// allocation is cudagraph-capture-safe (a raw hipMalloc/hipFree is illegal
// inside a stream capture and corrupts multi-size cudagraph capture).
// ----------------------------------------------------------------------------
AITER_C_ITFS void top_k_per_row_decode_fast_with_topk(
    aiter_tensor_t* logits,
    int64_t next_n,
    aiter_tensor_t* seqLens,
    aiter_tensor_t* indices,
    aiter_tensor_t* workspace,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    AITER_CHECK(topK == 512 || topK == 1024 || topK == 2048,
                "top_k_per_row_decode_fast_with_topk: topK must be 512, 1024, or 2048, got ",
                topK);

    TopKDecodeKernelArgsWithTopk args;
    size_t arg_size = sizeof(args);

    args.ptr_buf        = workspace->data_ptr();
    args.ptr_logits     = logits->data_ptr();
    args.ptr_seqLens    = seqLens->data_ptr();
    args.ptr_outIndices = indices->data_ptr();
    args.stride0        = static_cast<int32_t>(stride0);
    args.stride1        = static_cast<int32_t>(stride1);
    args.next_n         = static_cast<int32_t>(next_n);

    static AiterAsmKernel impl_K2048(
        "_ZN5aiter10DecodeTopKL10topKPerRowILi1024ELi2048ELi2048ELi512EEEvNS0_10TopKParamsE",
        "/topk_per_row_decode/asm_top_k_per_row_decode_with_topk.co");
    static AiterAsmKernel impl_K1024(
        "_ZN5aiter10DecodeTopKL10topKPerRowILi1024ELi2048ELi1024ELi512EEEvNS0_10TopKParamsE",
        "/topk_per_row_decode/asm_top_k_per_row_decode_with_topk.co");
    static AiterAsmKernel impl_K512(
        "_ZN5aiter10DecodeTopKL10topKPerRowILi1024ELi2048ELi512ELi512EEEvNS0_10TopKParamsE",
        "/topk_per_row_decode/asm_top_k_per_row_decode_with_topk.co");

    constexpr int kNumThreadsPerBlock = 1024;
    AITER_CHECK(numRows >> 31 == 0, "numRows too large: ", numRows);

    AiterAsmKernel& impl = (topK == 2048) ? impl_K2048
                          : (topK == 1024) ? impl_K1024
                          : impl_K512;
    impl.launch_kernel({&args,
                        &arg_size,
                        static_cast<int>(numRows),
                        1, 1,
                        kNumThreadsPerBlock,
                        1, 1,
                        stream});
}
