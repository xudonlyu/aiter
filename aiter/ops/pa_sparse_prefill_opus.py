# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based sparse paged prefill attention for DeepSeek-V4 on gfx950.

Two-region sparse scaled-dot-product attention over a paged prefix source
(``unified_kv``) and a flat per-fwd extend source (``kv``), with a per-head
softmax-denominator sink. The two regions share a single online-softmax
accumulator, making the order region-invariant.

The user-facing entry is :func:`pa_sparse_prefill_opus`; it forwards
to the JIT-compiled HIP kernel via
:func:`pa_sparse_prefill_opus_fwd`.

The kernel currently only compiles a single configuration:

* Head dim ``D == 512``.
* dtype ``bf16`` or ``fp16`` for Q/K/V/O; ``attn_sink`` is ``fp32``.
* Every entry in ``kv_indices_prefix`` / ``kv_indices_extend`` must be a
  valid row index into ``unified_kv`` / ``kv`` respectively. Empty CSR rows
  (``kv_indptr[i] == kv_indptr[i+1]``) are allowed.

See ``aiter/csrc/include/pa_sparse_prefill_opus.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_pa_sparse_prefill_opus"


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_opus_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_sparse_prefill_opus_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_sparse_prefill_opus_fake)
def pa_sparse_prefill_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources (paged ``unified_kv`` +
    flat per-fwd ``kv``), backed by the OPUS gfx950 HIP kernel.

    The trailing ``out`` keyword is an aiter-only convenience for callers
    that want to reuse a pre-allocated output buffer; pass ``None`` (the
    default) to have one allocated for you.

    Args:
      q:                 ``[T, H, D]`` bf16/fp16 query (T == N tokens).
      unified_kv:        ``[total_pages, D]`` prefix source (paged history).
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into
        ``unified_kv``, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, D]`` extend source (current fwd's
        just-computed K).
      kv_indices_extend: ``[total_extend]`` int32 row indices into ``kv``,
        concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, D]`` output buffer; allocated if
        ``None``.

    Returns:
      ``out`` (``[T, H, D]`` same dtype as ``q``).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_sparse_prefill_opus requires gfx950, got {gfx}")

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_sparse_prefill_opus expects fp16/bf16 q, got {q.dtype}")
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: unified_kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    pa_sparse_prefill_opus_fwd(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    return out


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_fp8_opus_fwd(
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
) -> None: ...


def _pa_sparse_prefill_fp8_opus_fake(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_sparse_prefill_fp8_opus_fake)
def pa_sparse_prefill_fp8_opus(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs.

    The trailing ``out`` keyword is an aiter-only convenience for callers that
    want to reuse a pre-allocated output buffer; pass ``None`` (the default) to
    have one allocated for you.

    Args:
      q_nope:            ``[T, H, 512]`` fp8 query without positional encoding.
      q_rope:            ``[T, H, 64]`` bf16 query RoPE encoding part.
      unified_kv_nope:   ``[total_pages, 512]`` fp8 prefix KV NoPE source.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix KV RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` fp8 extend KV NoPE source.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend KV RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the combined QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_sparse_prefill_fp8_opus requires gfx950, got {gfx}")

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if out is None:
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )

    pa_sparse_prefill_fp8_opus_fwd(
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
    )
    return out


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_fp8_opus_paged_fwd(
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
) -> None: ...


def _pa_sparse_prefill_fp8_opus_paged_fake(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(
    mutates_args=["out"], gen_fake=_pa_sparse_prefill_fp8_opus_paged_fake
)
def pa_sparse_prefill_fp8_opus_paged(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention reading vLLM's ``fp8_ds_mla`` paged pool directly.

    Same kernel and same numerics as :func:`pa_sparse_prefill_fp8_opus`, but the
    KV streams are addressed by slot index inside a page grid instead of being
    staged into a flat array first -- which is the whole point: it removes the
    dequantise-and-gather pass that would otherwise run over the pool on every
    call.

    A pool row is 576 B, 448 NoPE fp8 followed by 64 RoPE bf16, and the
    exponents are **not** in the row: every page ends with a tail of 8 B per
    token holding 7 per-64 UE8M0 block scales and a pad byte.  A per-64 exponent
    applies exactly to both of its per-32 halves, with the identical ``exp+127``
    encoding, so the kernel consumes the pool's scales unchanged -- there is no
    requantisation, no in-place rewrite of the cache, and no global exponent
    frame to keep in step.

    Args:
      q_nope:            ``[N, H, 512]`` fp8, pre-packed (448 NoPE + 14 per-32
        E8M0 + pad), exactly as the flat op wants it.
      q_rope:            ``[N, H, 64]`` bf16.
      unified_kv_nope:   ``[rows, 512]`` fp8 view of the prefix pool, row stride
        576.  Its last 64 columns overlap the row's RoPE half and are never
        used, so a plain strided view of the pool is what to pass.
      unified_kv_rope:   ``[rows, 64]`` bf16 view of the same pool, based at
        byte 448 of row 0, row stride 288.
      kv_nope, kv_rope:  the extend segment's views, same shapes and strides.
      kv_indices_*:      ``[nnz]`` int32 slot ids, concatenated per token.
      kv_indptr_*:       ``[N+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denominator bias.
      page_shift_*:      ``page_size == 1 << page_shift``.
      rows_per_page_*:   ``bytes_per_page / 576``.
      scale_off_*:       byte offset of the page's scale tail.
      out:               optional ``[N, H, 512]`` bf16 buffer.

    The prefix and extend segments may live in different pools with different
    page sizes, hence one descriptor each.

    Requires ``H > 32``: only the T_M=8 pipeline has a paged variant.  There is
    also no flat fallback -- ``page_shift == 0`` is rejected rather than treated
    as a degenerate page grid, because a flat row carries 14 per-32 exponents
    inline and this kernel always fetches 7 per-64 ones from a page tail.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(
            f"pa_sparse_prefill_fp8_opus_paged requires gfx950, got {gfx}"
        )
    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if h <= 32:
        raise RuntimeError(
            f"pa_sparse_prefill_fp8_opus_paged is only compiled for H > 32, got H={h}"
        )
    if out is None:
        out = torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)
    elif out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )

    pa_sparse_prefill_fp8_opus_paged_fwd(
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
    )
    return out


__all__ = [
    "pa_sparse_prefill_fp8_opus",
    "pa_sparse_prefill_fp8_opus_paged",
    "pa_sparse_prefill_fp8_opus_paged_fwd",
    "pa_sparse_prefill_fp8_opus_fwd",
    "pa_sparse_prefill_opus",
    "pa_sparse_prefill_opus_fwd",
]
