# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Probe: what does a wider DSpark block cost the routed MoE half?

DSpark's block size is 5 today, so the draft proposes 5 tokens and the verify runs 6
rows. Block size 10 proposes 10 and verifies 11. This measures the *MoE* side of that
trade at the routing each side actually sees, which is the cheap half of the question --
the MoE is latency bound and adds ~0.21 us a distinct expert:

  verify  6 rows, top-6 of 384, 192 experts a rank   (measured: 15.11 us)
  verify 11 rows, top-6 of 384, 192 experts a rank
  draft   5 rows, top-3 of 128, 64 experts a rank
  draft  10 rows, top-3 of 128, 64 experts a rank

Whether the extra positions are *accepted* is a checkpoint question that cannot be
measured here: it needs the server restarted with --speculative-dspark-block-size 10.

Each shape runs in its own process because the bench caches compiled kernels and layer
buffer shapes at module scope; ``--case`` is the child's half of that handshake.

  python benchmark/probe_deepseek_v41_rows.py            # parent, runs every case
  python benchmark/probe_deepseek_v41_rows.py --case verify11 --local 192 ...
"""

import os
import subprocess
import sys

# (case name, rows, top_k, experts a rank, distinct experts to route)
# ``distinct`` stays at 53 % of the slot count on every row, which is what the shipped
# verify measures (19 distinct for 6 x 6 = 36 slots). Keep it well under the slot count:
# ``routing`` rejection-samples until every requested expert appears, so a distinct close
# to the slot count spins forever (5 x 3 = 15 slots cannot reliably cover 14 experts).
CASES = [
    ("verify6", 6, 6, 192, 19),
    ("verify11", 11, 6, 192, 30),
    ("draft5", 5, 3, 64, 8),
    ("draft10", 10, 3, 64, 16),
]
LABELS = {
    "verify6": "verify  6 rows (block 5, shipped)",
    "verify11": "verify 11 rows (block 10)",
    "draft5": "draft   5 rows (block 5, shipped)",
    "draft10": "draft  10 rows (block 10)",
}


def child(case: str, side: list[int], repeats: int, replays: int, offset: int) -> None:
    name, rows, top_k, local, distinct = next(c for c in CASES if c[0] == case)
    import bench_deepseek_v41_routed_moe as B

    # set before Bench() is constructed: buffers, route words and the routing
    # generator all read these at module scope
    B.ROWS, B.TOP_K = rows, top_k
    sys.argv = [
        "probe_rows",
        "--distinct", str(distinct),
        "--variants", "pipe_clc",
        "--side", *[str(s) for s in side],
        "--repeats", str(repeats),
        "--replays", str(replays),
        "--local-experts", str(local),
        "--offset", str(offset),
    ]
    B.main()


def parent(side: list[int], repeats: int, replays: int, offset: int, only: str) -> None:
    for name, rows, top_k, local, _ in CASES:
        if only and only not in name:
            continue
        print(f"### {LABELS[name]}  top-{top_k} of 384, {local} experts a rank", flush=True)
        child_env = dict(os.environ)
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import probe_deepseek_v41_rows as P; "
            "P.child(%r, %r, %d, %d, %d)"
            % (
                os.path.dirname(os.path.abspath(__file__)),
                name, side, repeats, replays, offset,
            )
        )
        r = subprocess.run([sys.executable, "-c", code], env=child_env)
        if r.returncode:
            print(f"### {name} FAILED with {r.returncode}", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--side", type=int, nargs="+", default=[0])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--only", default="")
    p.add_argument("--case", default="", help="internal: run one case in this process")
    args = p.parse_args()
    if args.case:
        child(args.case, args.side, args.repeats, args.replays, args.offset)
        return
    parent(args.side, args.repeats, args.replays, args.offset, args.only)


if __name__ == "__main__":
    main()
