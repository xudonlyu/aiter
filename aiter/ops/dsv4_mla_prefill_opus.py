# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prebuilt-code-object build of the DSv4 MLA paged sparse prefill.

Same kernel as ``aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus_paged``
-- the JIT twin -- shipped as ``hsa/gfx950/dsv4_mla_prefill_opus/*.co`` so a
deployment compiles only the host-side loader, not the kernel.  The two are
required to be bit-identical; ``op_tests/test_dsv4_mla_prefill_opus.py`` asserts
it with ``torch.equal`` rather than a tolerance, because a tolerance would pass a
code object built from different source.

Import one module or the other, never both -- they register the same op under
different names, and only the JIT one can be rebuilt from this checkout.
"""

from typing import Optional

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

_MODULE = "module_dsv4_mla_prefill_opus"

# Part of the op's contract rather than of this build, so a caller can gate on
# it without importing the kernel header.  H <= 32 is a different pipeline
# (T_M=1) and has no paged variant; page_shift 0 is not a flat fallback.
DSV4_MLA_PREFILL_OPUS_MIN_H = 33
DSV4_MLA_PREFILL_OPUS_ROW_BYTES = 576


@compile_ops(_MODULE, fc_name="dsv4_mla_prefill_opus_warmup", ffi_type="ctypes")
def _warmup() -> int: ...


@compile_ops(_MODULE, fc_name="dsv4_mla_prefill_opus_fwd", ffi_type="ctypes")
def _paged_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    page_shift_prefix: int,
    rows_per_page_prefix: int,
    scale_off_prefix: int,
    page_shift_extend: int,
    rows_per_page_extend: int,
    scale_off_extend: int,
    kv_lens_prefix: torch.Tensor,
    kv_lens_extend: torch.Tensor,
    kv_stride_q_prefix: int,
    kv_stride_q_extend: int,
) -> int: ...


def _empty_i32(like: torch.Tensor) -> torch.Tensor:
    """Sentinel for an omitted optional tensor argument (numel 0 == not given)."""
    return torch.empty(0, dtype=torch.int32, device=like.device)


def _fake(
    q_nope: torch.Tensor,
    *args,
    out: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_fake)
def dsv4_mla_prefill_opus(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    page_shift_prefix: int,
    rows_per_page_prefix: int,
    scale_off_prefix: int,
    page_shift_extend: int,
    rows_per_page_extend: int,
    scale_off_extend: int,
    out: Optional[torch.Tensor] = None,
    kv_lens_prefix: Optional[torch.Tensor] = None,
    kv_lens_extend: Optional[torch.Tensor] = None,
    kv_stride_q_prefix: int = 0,
    kv_stride_q_extend: int = 0,
) -> torch.Tensor:
    """DSv4 sparse prefill reading vLLM's ``fp8_ds_mla`` paged pool directly.

    See :func:`aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus_paged`
    for the argument contract -- this is the same op, same signature, backed by a
    prebuilt code object instead of a JIT build.

    In short: a pool row is 576 B (448 NoPE fp8 | 64 RoPE bf16) and the E8M0
    exponents sit out of line in each page's tail, 8 B per token holding 7 per-64
    UE8M0 scales.  A per-64 exponent applies exactly to both of its per-32
    halves, so the pool's scales are consumed unchanged -- no requantisation, no
    in-place rewrite, no global exponent frame.  ``q_nope`` arrives as plain
    bf16 ``[N, H, 448]`` and is packed in the kernel prologue.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"dsv4_mla_prefill_opus requires gfx950, got {gfx}")
    if q_nope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_nope must be bf16 [N, H, 448], got {q_nope.dtype}")
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")
    if unified_kv_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            "unified_kv_nope/kv_nope dtype mismatch: "
            f"{unified_kv_nope.dtype}, {kv_nope.dtype}"
        )

    t, h = q_nope.shape[0], q_nope.shape[1]
    if h < DSV4_MLA_PREFILL_OPUS_MIN_H:
        raise RuntimeError(
            f"dsv4_mla_prefill_opus is only compiled for H >= "
            f"{DSV4_MLA_PREFILL_OPUS_MIN_H}, got H={h}"
        )
    if out is None:
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )

    _paged_fwd(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
        int(page_shift_prefix),
        int(rows_per_page_prefix),
        int(scale_off_prefix),
        int(page_shift_extend),
        int(rows_per_page_extend),
        int(scale_off_extend),
        _empty_i32(q_nope) if kv_lens_prefix is None else kv_lens_prefix,
        _empty_i32(q_nope) if kv_lens_extend is None else kv_lens_extend,
        int(kv_stride_q_prefix),
        int(kv_stride_q_extend),
    )
    return out


# Registering a code object is illegal while a stream is capturing, and with
# cudagraphs the first call can land inside one.  Force it here, at import.
_warmup()

__all__ = [
    "DSV4_MLA_PREFILL_OPUS_MIN_H",
    "DSV4_MLA_PREFILL_OPUS_ROW_BYTES",
    "dsv4_mla_prefill_opus",
]
