# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probe: does the routed pair pay for MMA instruction *count*, or only for TMA rounds?

This decides whether a half-width (128-N) GEMM1 tile is viable. 128 N keeps the same
FLOPs, halves the A bytes a K chunk needs -- so a 1024-K chunk fits the 232448-byte smem
budget -- but it doubles the number of ``tcgen05`` MMA instructions for the same arithmetic.
If MMA issue is free the trade is a pure win; if not, the 4.1 % the chunk count buys is at
risk against it.

The probe holds TMA traffic and every barrier fixed and only varies how many of the four
MMA sub-blocks in a K chunk actually issue. ``i_desc`` >> 29 is the sub-block index (the
four shipped instructions are 0x08820280, 0x28820290, 0x488202a0, 0x688202b0), so the
helper returns early for ``sub >= --subs``. The commit and the mbarrier arrives sit outside
the MMA block, so the pipeline shape does not change. **The output is wrong on purpose.**

Reading it:

  flat across --subs   MMA issue is free; 128 N costs nothing and the chunk win is real
  scales with --subs   MMA issue is a real term; subtract it from the 4.1 % before building

  python benchmark/probe_deepseek_v41_gemm1_mma.py --subs 0 1 2 4 --distinct 19
"""

import re
import sys

import bench_deepseek_v41_routed_moe as B
from sgl_nvidia_mega.deepseek_v41.sm103.moe import routed_expert_gemm as reg

SRC = (reg.SOURCE_DIR / "gemm1.cu").read_text()

# The helper's body, so a probe can insert its early return.
HELPER = re.compile(
    r"(tcgen05_mma_mxf8_bs\(int taddr, uint64_t a_desc, uint64_t b_desc,\s*"
    r"uint32_t i_desc, int sfa_taddr, int sfb_taddr,\s*int enable_input_d\) \{)"
)


def patch(subs: int) -> str:
    """gemm1.cu with only the first ``subs`` of the four MMA sub-blocks issuing."""
    if subs >= 4:
        return SRC
    body, n = HELPER.subn(rf"\1\n  if ((i_desc >> 29) >= {subs}U) {{ return; }}", SRC)
    if n != 1:
        raise RuntimeError(f"expected one tcgen05_mma_mxf8_bs definition, patched {n}")
    return body


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subs", type=int, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--distinct", type=int, nargs="+", default=[19])
    p.add_argument("--side", type=int, nargs="+", default=[0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--local-experts", type=int, default=192)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    extra = {f"gemm1_m{n}": patch(n) for n in args.subs}
    orig_load, orig_grids = B.load_kernels, B.grids

    def load_kernels(device_index, sources=None):
        merged = dict(extra)
        merged.update(sources or {})
        return orig_load(device_index, merged)

    def grids(variant, ctas, sms):
        m = re.fullmatch(r"m(\d+)_pipe_clc", variant)
        if m:
            return (f"gemm1_m{m.group(1)}", (reg.GEMM1_UNITS, ctas, 1)), (
                "gemm2_pipe",
                (reg.GEMM2_UNITS, ctas, 1),
            )
        return orig_grids(variant, ctas, sms)

    B.load_kernels, B.grids = load_kernels, grids
    sys.argv = [
        "probe_gemm1_mma",
        "--distinct", *[str(d) for d in args.distinct],
        "--variants", *[f"m{n}_pipe_clc" for n in args.subs],
        "--side", *[str(s) for s in args.side],
        "--repeats", str(args.repeats),
        "--replays", str(args.replays),
        "--local-experts", str(args.local_experts),
        "--offset", str(args.offset),
    ]
    B.main()


if __name__ == "__main__":
    main()
