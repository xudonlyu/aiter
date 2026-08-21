# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The prebuilt code object against its JIT twin, on identical inputs.

Required to be **bit**-identical, not close.  A tolerance here would pass a .co
built from different source, which is the only failure this test exists to
catch.
"""
import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_pa_sparse_prefill_fp8_opus_paged import (  # noqa: E402
    build_pool, quant_per64, D, D_NOPE, D_ROPE, _skip_if_unsupported)

from aiter.ops import dsv4_mla_prefill_opus as co  # noqa: E402
from aiter.ops import pa_sparse_prefill_opus as jit  # noqa: E402


def _case(T, H, rows, plen, elen, page_size, dense):
    dev = torch.device("cuda")
    torch.manual_seed(20260821)
    q = (torch.randn(T, H, D, device=dev) * .5).to(torch.bfloat16)
    kv = (torch.randn(rows, D, device=dev) * .5).to(torch.bfloat16)
    sink = torch.randn(H, dtype=torch.float32, device=dev) * .25
    nope_fp8, enc, _ = quant_per64(kv[:, :D_NOPE])
    rope = kv[:, D_NOPE:].contiguous()
    _, nv, rv, g = build_pool(nope_fp8, enc, rope, page_size)
    p_ix = torch.randint(0, rows, (T, plen), dtype=torch.int32, device=dev)
    e_ix = torch.randint(0, rows, (T, elen), dtype=torch.int32, device=dev)
    p_ip = torch.arange(0, (T + 1) * plen, plen, dtype=torch.int32, device=dev)
    e_ip = torch.arange(0, (T + 1) * elen, elen, dtype=torch.int32, device=dev)
    args = (q[..., :D_NOPE].contiguous(), q[..., D_NOPE:].contiguous(), nv, rv,
            p_ix.reshape(-1), p_ip, nv, rv, e_ix.reshape(-1), e_ip, sink,
            1.0 / math.sqrt(D), g["page_shift"], g["rows_per_page"],
            g["scale_off"], g["page_shift"], g["rows_per_page"], g["scale_off"])
    kw = {}
    if dense:
        kw = dict(kv_lens_prefix=torch.full((T,), plen, dtype=torch.int32, device=dev),
                  kv_lens_extend=torch.full((T,), elen, dtype=torch.int32, device=dev),
                  kv_stride_q_prefix=plen, kv_stride_q_extend=elen)
    a = jit.pa_sparse_prefill_fp8_opus_paged(*args, **kw)
    b = co.dsv4_mla_prefill_opus(*args, **kw)
    torch.cuda.synchronize()
    return a, b


def main():
    if _skip_if_unsupported():
        return
    shapes = [(1024, 128, 8192, 128, 512, 64),
              (512, 128, 8192, 128, 1024, 64),
              (256, 192, 8192, 128, 256, 128),
              (512, 64, 8192, 128, 32, 32)]
    bad = 0
    for T, H, rows, plen, elen, ps in shapes:
        for dense in (False, True):
            a, b = _case(T, H, rows, plen, elen, ps, dense)
            same = torch.equal(a, b)
            nf = int((~torch.isfinite(b)).sum())
            bad += (not same) or nf
            form = "dense" if dense else "csr  "
            print(f"T={T:>5} H={H:>4} rows/tok={plen+elen:>5} ps={ps:>4} {form} "
                  f"nonfinite={nf:>3}  co == jit: {same}"
                  f"{'' if same and not nf else '   <-- FAIL'}", flush=True)
    print("\nFAILURES:" if bad else "\nthe code object is bit-identical to the JIT build", bad or "")


if __name__ == "__main__":
    main()
