# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probe: how much of the routed pair is GEMM1's K-chunk chain?

Patches only the K-loop bounds of ``gemm1.cu`` (every ``iter_k* < 10`` and the
``iter_k_4 + 1 == 10`` last-iteration test) to ``--chunks`` so GEMM1 reads that many of
its ten 512-K chunks and accumulates garbage. **The output is wrong on purpose.** The
timing is still readable because at D=1 the half reads 2.3 MB, whose roofline is 0.3 us,
so a microsecond-scale change is the latency chain and not bandwidth.

The measured shape (one GPU, ``--side 0``, pipe_clc for GEMM2) is linear in the chunk
count, and the intercept is GEMM2:

    D=19   t = 9.89 + 0.528 * chunks
    D=1    t = 6.50 + 0.494 * chunks

So halving the chunk count (10 -> 5) is worth 2.6 us a half and no more, and the K rounds
are ~35 % of the half. See docs/findings/deepseek_v41_bs1_attention.md.

  python benchmark/probe_deepseek_v41_gemm1_chunks.py --chunks 2 5 7 10 --distinct 1 19
"""

import re
import sys

import bench_deepseek_v41_routed_moe as B
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

SRC = (reg.SOURCE_DIR / "gemm1.cu").read_text()


def patch(n: int) -> str:
    """gemm1.cu with every K-loop bound set to ``n`` chunks."""
    out = re.sub(r"(_?iter_k(?:_\d)?) < 10;", lambda m: f"{m.group(1)} < {n};", SRC)
    out = out.replace("if (iter_k_4 + 1 == 10)", f"if (iter_k_4 + 1 == {n})")
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--chunks", type=int, nargs="+", default=[2, 5, 7, 10])
    p.add_argument("--distinct", type=int, nargs="+", default=[1, 19])
    p.add_argument("--side", type=int, nargs="+", default=[0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--local-experts", type=int, default=192)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    extra = {f"gemm1_k{n}": patch(n) for n in args.chunks}
    orig_load, orig_grids = B.load_kernels, B.grids

    def load_kernels(device_index, sources=None):
        merged = dict(extra)
        merged.update(sources or {})
        return orig_load(device_index, merged)

    def grids(variant, ctas, sms):
        m = re.fullmatch(r"k(\d+)_(pipe_clc|clc)", variant)
        if m:
            n, rest = m.group(1), m.group(2)
            gemm2 = "gemm2_pipe" if rest == "pipe_clc" else "gemm2"
            return (f"gemm1_k{n}", (reg.GEMM1_UNITS, ctas, 1)), (
                gemm2,
                (reg.GEMM2_UNITS, ctas, 1),
            )
        return orig_grids(variant, ctas, sms)

    B.load_kernels, B.grids = load_kernels, grids
    sys.argv = [
        "probe_gemm1_chunks",
        "--distinct", *[str(d) for d in args.distinct],
        "--variants", *[f"k{n}_pipe_clc" for n in args.chunks],
        "--side", *[str(s) for s in args.side],
        "--repeats", str(args.repeats),
        "--replays", str(args.replays),
        "--local-experts", str(args.local_experts),
        "--offset", str(args.offset),
    ]
    B.main()


if __name__ == "__main__":
    main()
