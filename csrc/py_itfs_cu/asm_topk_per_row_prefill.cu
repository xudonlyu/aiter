// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_hip_common.h"

struct __attribute__((packed)) TopKPrefillKernelArgs
{
    void* ptr_workspace;
    void* ptr_logits;
    void* ptr_rowStarts;
    void* ptr_rowEnds;
    void* ptr_indices;
    void* ptr_values;
    int32_t stride0;
    int32_t stride1;
};

AITER_C_ITFS void top_k_per_row_prefill_fast(
    aiter_tensor_t* logits,
    aiter_tensor_t* rowStarts,
    aiter_tensor_t* rowEnds,
    aiter_tensor_t* indices,
    aiter_tensor_t* values,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    constexpr int kTopK = 2048;
    int64_t workspace_size = kTopK * (sizeof(float) + sizeof(int32_t)) * numRows;
    void* workspace = nullptr;
    HIP_CALL(hipMalloc(&workspace, workspace_size));

    TopKPrefillKernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_workspace = workspace;
    args.ptr_logits    = logits->data_ptr();
    args.ptr_rowStarts = rowStarts->data_ptr();
    args.ptr_rowEnds   = rowEnds->data_ptr();
    args.ptr_indices   = indices->data_ptr();
    args.ptr_values    = (values != nullptr) ? values->data_ptr() : nullptr;
    args.stride0       = static_cast<int32_t>(stride0);
    args.stride1       = static_cast<int32_t>(stride1);

    static AiterAsmKernel impl_topk_prefill(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi2048ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill.co");

    constexpr int kNumThreadsPerBlock = 1024;
    AITER_CHECK(numRows >> 31 == 0, "numRows too large: ", numRows);

    impl_topk_prefill.launch_kernel({&args,
                                     &arg_size,
                                     static_cast<int>(numRows),
                                     1, 1,
                                     kNumThreadsPerBlock,
                                     1, 1,
                                     stream});

    HIP_CALL(hipFree(workspace));
}

// ----------------------------------------------------------------------------
// `top_k_per_row_prefill_fast_with_topk` — runtime-K variant of the asm-fast
// prefill top-K kernel. Loads
// `topk_per_row_prefill/asm_top_k_per_row_prefill_with_topk.co` (built with
// K=512/1024/2048 explicit instantiations) and dispatches to the matching
// mangled symbol based on `topK`. This is the path used on gfx950 (FULL cudagraph)
// where the base `top_k_per_row_prefill_fast` .co is not shipped.
//
// The per-call scratch buffer (`topK * (sizeof(float) + sizeof(int32_t)) *
// numRows` bytes) is allocated by the caller (Python, via the torch caching
// allocator) and passed in as `workspace`, so this TU stays torch-free and the
// allocation is cudagraph-capture-safe (a raw hipMalloc/hipFree is illegal
// inside a stream capture and corrupts multi-size cudagraph capture).
//
// Constraints: `topK` must be one of {512, 1024, 2048}.
// ----------------------------------------------------------------------------
AITER_C_ITFS void top_k_per_row_prefill_fast_with_topk(
    aiter_tensor_t* logits,
    aiter_tensor_t* rowStarts,
    aiter_tensor_t* rowEnds,
    aiter_tensor_t* indices,
    aiter_tensor_t* values,
    aiter_tensor_t* workspace,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t topK,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    AITER_CHECK(topK == 512 || topK == 1024 || topK == 2048,
                "top_k_per_row_prefill_fast_with_topk: topK must be 512, 1024, or 2048, got ",
                topK);

    TopKPrefillKernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_workspace = workspace->data_ptr();
    args.ptr_logits    = logits->data_ptr();
    args.ptr_rowStarts = rowStarts->data_ptr();
    args.ptr_rowEnds   = rowEnds->data_ptr();
    args.ptr_indices   = indices->data_ptr();
    args.ptr_values    = (values != nullptr) ? values->data_ptr() : nullptr;
    args.stride0       = static_cast<int32_t>(stride0);
    args.stride1       = static_cast<int32_t>(stride1);

    // Three pre-instantiated K variants; the mangled name differs only in the
    // third template integer (`Li2048E` / `Li1024E` / `Li512E`).
    static AiterAsmKernel impl_K2048(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi2048ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill_with_topk.co");
    static AiterAsmKernel impl_K1024(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi1024ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill_with_topk.co");
    static AiterAsmKernel impl_K512(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi512ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill_with_topk.co");

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
