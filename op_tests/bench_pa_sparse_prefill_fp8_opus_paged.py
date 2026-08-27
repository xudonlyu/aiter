#!/usr/bin/env python3
"""Per-call cost of the paged opus fp8 prefill against what it replaces.

Arms, at the DSv4-Pro layer shapes:

  triton      vLLM's Triton sparse prefill -- what the server runs today
  flat        aiter flat fp8 opus, plus the Q pack pass it needs
  flat-kern   the same kernel with the pack excluded, for reference
  paged       the new op: reads the pool directly, packs Q itself

`paged` vs `flat` is what fusing the pack and dropping the staged array
buys at the op level.  `paged` vs `triton` is the number that matters to
the server, and it excludes vLLM's dequantise-and-gather entirely, which
the paged op removes but the Triton arm here never pays either (its KV is
already a bf16 array), so it is if anything conservative.
"""
import argparse, math, statistics, torch
from paged_accuracy import (quant_per64, build_pool, pack_flat, pack_q,
                            D, D_NOPE, D_ROPE)
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_fp8_opus, pa_sparse_prefill_fp8_opus_paged)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_prefill


def timed(fn, warmup, groups, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(groups):
        b, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        b.record()
        for _ in range(inner):
            fn()
        e.record(); e.synchronize()
        v.append(b.elapsed_time(e) * 1000 / inner)
    return statistics.median(sorted(v))


def run(T, H, rows, plen, elen, ps, warmup, groups, inner):
    dev = torch.device("cuda"); torch.manual_seed(20260821)
    q = (torch.randn(T, H, D, device=dev) * .5).to(torch.bfloat16)
    kv = (torch.randn(rows, D, device=dev) * .5).to(torch.bfloat16)
    sink = torch.randn(H, dtype=torch.float32, device=dev) * .25
    scale = 1.0 / math.sqrt(D)
    nope_fp8, enc, _ = quant_per64(kv[:, :D_NOPE])
    rope = kv[:, D_NOPE:].contiguous()
    pool, nv, rv, geom = build_pool(nope_fp8, enc, rope, ps)
    flat_nope = pack_flat(nope_fp8, enc)
    p_ix = torch.randint(0, rows, (T, plen), dtype=torch.int32, device=dev)
    e_ix = torch.randint(0, rows, (T, elen), dtype=torch.int32, device=dev)
    p_ip = torch.arange(0, (T+1)*plen, plen, dtype=torch.int32, device=dev)
    e_ip = torch.arange(0, (T+1)*elen, elen, dtype=torch.int32, device=dev)
    q_rope = q[..., D_NOPE:].contiguous()
    q_nope_bf16 = q[..., :D_NOPE].contiguous()
    q_nope_fp8 = pack_q(q_nope_bf16.reshape(-1, D_NOPE)).reshape(T, H, 512)
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=dev)
    tri_ix = torch.cat((p_ix, e_ix), dim=1)
    tri_len = torch.full((T,), plen+elen, dtype=torch.int32, device=dev)
    p_len = torch.full((T,), plen, dtype=torch.int32, device=dev)
    e_len = torch.full((T,), elen, dtype=torch.int32, device=dev)

    def triton():
        rocm_sparse_attn_prefill(q=q, kv=kv.unsqueeze(1), indices=tri_ix,
                                 topk_length=tri_len, scale=scale, head_dim=D,
                                 nope_head_dim=D_NOPE, rope_head_dim=D_ROPE,
                                 attn_sink=sink, output=out)
    def flat_kern():
        pa_sparse_prefill_fp8_opus(q_nope_fp8, q_rope, flat_nope, rope,
                                   p_ix.reshape(-1), p_ip, flat_nope, rope,
                                   e_ix.reshape(-1), e_ip, sink, scale, out=out)
    def flat_full():
        # NOTE: pack_q here is the torch *reference* packer, not aiter's
        # pa_fp8_q_pack kernel, so this arm overstates the flat path badly
        # (~850 us vs the kernel's measured 33 us at T=1024 H=128).  The real
        # comparison is flat_kern + 33 us; this arm is kept only as an upper
        # bound.  pa_fp8_q_pack lives on the h40 branch, not on this base.
        pack_q(q_nope_bf16.reshape(-1, D_NOPE))
        flat_kern()
    def paged():
        pa_sparse_prefill_fp8_opus_paged(
            q_nope_bf16, q_rope, nv, rv, p_ix.reshape(-1), p_ip, nv, rv,
            e_ix.reshape(-1), e_ip, sink, scale, geom["page_shift"],
            geom["rows_per_page"], geom["scale_off"], geom["page_shift"],
            geom["rows_per_page"], geom["scale_off"], out=out,
            kv_lens_prefix=p_len, kv_lens_extend=e_len,
            kv_stride_q_prefix=plen, kv_stride_q_extend=elen)

    arms = [("triton", triton), ("flat_kern", flat_kern),
            ("flat_full", flat_full), ("paged", paged)]
    lat = {}
    for _ in range(2):                       # interleave
        for n, f in arms:
            r = timed(f, warmup, groups, inner)
            if n not in lat or r < lat[n]:
                lat[n] = r
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=128)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--groups", type=int, default=9)
    ap.add_argument("--inner", type=int, default=4)
    a = ap.parse_args()
    cases = [("hca-4k", 128, 32), ("hca-50k", 128, 400),
             ("csa", 128, 512), ("hca-128k", 128, 1024)]
    print(f"T={a.tokens} H={a.heads} page_size={a.page_size} gpu={torch.cuda.get_device_name()}")
    print(f"{'case':10}{'rows':>6}{'triton':>10}{'flat_kern':>11}{'flat_full':>11}{'paged':>10}"
          f"{'paged/triton':>14}{'paged/flat_full':>17}{'paged/flat_kern':>17}")
    for name, p, e in cases:
        L = run(a.tokens, a.heads, a.rows, p, e, a.page_size, a.warmup, a.groups, a.inner)
        print(f"{name:10}{p+e:>6}{L['triton']:>10.1f}{L['flat_kern']:>11.1f}"
              f"{L['flat_full']:>11.1f}{L['paged']:>10.1f}"
              f"{L['triton']/L['paged']:>13.2f}x{L['flat_full']/L['paged']:>16.3f}x"
              f"{L['flat_kern']/L['paged']:>16.3f}x", flush=True)


if __name__ == "__main__":
    main()
