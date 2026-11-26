# DeepSeek-V4.1-Flash, batch 1: attention slots on GB300

Measurements of the attention slots at batch 1 on the layout this package serves for
narrow steps: attention TP8 (8 local heads, 1 local group) with the MoE at TP4 x EP2
(`--tp 8 --ep-size 2`, 192 routed experts an EP rank). Everything below is served
`/v1/completions` decoding of one request; the per-step times are the step stamps the
plugin writes in-graph (`SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES`), not client gaps.

## How these numbers were taken

| Setting | Value | Why |
| --- | --- | --- |
| DSpark | `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` | The shipped batch-1 spec setup; each step is a 6-row verify half and a draft half, reported separately |
| Acceptance | `SGLANG_SIMULATE_ACC_LEN=3.51`, `match-expected`, `real-draft-token` | measurement-only: holds the verify width at 6 rows in every arm, so step times compare; it cannot show accept-rate effects |
| Routing | `SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING=synth` | measurement-only: removes expert-selection jitter between arms |
| Prompts | random token ids, `benchmark/static_table.py --isl 8192 --osl 512 --conc 1` | Seed-stable across arms and pass orders |
| Passes | warm-up, then one forward and one `--reverse` pass | The mean of the two passes is reported |

The two measurement-only settings are what make an arm-to-arm step time comparable at
the few-microsecond level. They also bound what the numbers mean: with synthetic routing
and random tokens these step times say how much faster the kernel chain is, not what a
real workload's end-to-end throughput becomes (see [Real prompts](#real-prompts-bench32)
for that direction).

## Baseline reproducibility

Three launches of the same slot list, same binary, GPUs empty before each launch
(`nvidia-smi` showed no compute processes), gave:

| Launch | Step period median | Verify half | Draft half |
| --- | --- | --- | --- |
| 1 | 2909.2 us | 2618.4 us | 291.3 us |
| 2 | 2907.2 us | 2616.6 us | 290.1 us |
| 3 (not measured) | — | — | — |

2.0 us apart, with the per-step p10/p90 inside a launch spanning about 14 us. That is the
noise floor the deltas below are read against. Two launches taken earlier in the session
read 3262.5 us and 3278.9 us with a byte-identical slot list; they did not reproduce and
are not used as a baseline. What they show is that this node's step time can sit in a
different regime across launches (another job of the same user started on the same 8
GPUs mid-session), so an arm is only compared against baselines taken in the same state,
and every pass records the number of GPU compute processes.

## `attn_core_lean`: measured under fixed routing

Read the number below as a kernel-chain bound; the live-router gain is smaller and is
what the next section reports.

Same slot list as the baseline plus `attn_core_lean`, on the KV layout it requires
(`SGLANG_DSV4_KV_LAYOUT=v41_new`, 528-byte window rows and 288-byte FP4 compressed rows):

| Arm | Step period median | Verify half | Draft half | Step p50 | tok/s/user p50 |
| --- | --- | --- | --- | --- | --- |
| baseline | 2908.2 us (mean of 2) | 2617.5 us | 290.7 us | 2.908 ms | 1204 |
| `attn_core_lean` | 2724.8 us | 2443.6 us | 281.3 us | 2.722 ms | 1283 |
| difference | **-183.4 us (-6.3 %)** | **-173.9 us** | **-9.4 us** | -0.186 ms (-6.4 %) | +6.6 % |

Both arms: 3 passes, 3000+ verify steps each, 8 GPU compute processes through every pass.

The verify half accounts for 174 of the 183 us: the one-launch core replaces swap-AB
*and* the separate combine kernel, and applies the output's inverse RoPE inside the
kernel, so the fused `wo_a` consumes an identity rotation. The draft half keeps its
8 us: its window attention runs the same core's 3-split variant.

The model page's "about 130 us per verify step" was measured on an earlier stack; this
run measures 174 us under fixed routing. The conditions differ, so the two are recorded
against their own setups rather than reconciled.

Engagement is logged, so a silently-fallback arm cannot be read as a win:

```
NVIDIA megakernel: slot attn_core_lean serves the target sparse attention at 6 rows
  (first layer 2, window indices before the PDL wait, ...)
NVIDIA megakernel: slot attn_core_lean serves the target window attention at 6 rows
  (first layer 0, ...)
NVIDIA megakernel: swap-AB declines: the window KV cache holds 528 bytes per token
  (swap-AB reads the V4 layout's 584)
```

### Accuracy

`test/kernels/test_deepseek_v41_core_lean.py` passes (6 tests), including
`test_core_lean_against_fp64`, which bounds the kernel against an fp64 reference at
relative RMS below 4e-3, and the decline conditions (rows > 8, heads != 8, a 1024-byte
row, a V4 pool).

This is a **numerics change**, not a bitwise one: the KV rows are the V4.1 encodings,
P is BF16 without compensation, and the split and merge follow muLLM's 64-key order.

## `attn_cluster_merge`: not resolved

| Arm | Step period median |
| --- | --- |
| baseline (that state) | 3262.5 us |
| `+attn_cluster_merge` | 3278.9 us |

+16.4 us, one launch per arm, in the machine state that later failed to reproduce. At
attention TP8 the eight local heads are exactly the heads the cluster merge has to
exchange, so the DSM exchange costs about what the combine kernel it removes costs. The
packaged microbenchmark's -3.3 us was one layer in isolation. Re-measurement in a
reproducible state is needed before calling this a gain or a loss; it is not a win on
the evidence so far.

## Real routing and real prompts: the gain is smaller

The numbers above hold the routing fixed to remove arm-to-arm jitter. Running the three
arms again with the router live (`SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING=0`,
everything else unchanged) and, per arm, also over 8 real bench32 rows:

| Arm | Period median | Verify half | Draft half | bench32 step p50 | bench32 step p99 | TTFT p50 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 2775.1 us | 2521.9 us | 271.1 us | 2.891 ms | 4.121 ms | 400.6 ms |
| `attn_core_lean` | 2727.8 us | 2447.7 us | 280.3 us | 2.842 ms | 3.753 ms | 386.7 ms |
| `+attn_wo_a_fp8`, `+attn_index_weights_bf16` | 2738.2 us | 2462.7 us | 276.4 us | 2.850 ms | 3.730 ms | 387.2 ms |

**The fixed-routing -183 us overstates the real-routing gain.** With the router live the
period moves 47 us and the client's own step p50 moves 49 us, not 183. The whole
difference is in the baseline: its verify half reads 2617.5 us under fixed routing and
2521.9 us with the router live, while `attn_core_lean`'s reads 2443.6 and 2447.7 us — the
core does not care, the baseline does. `synth` defaults to 19 distinct target experts per
layer, dealt `half0=9` to one EP half and 10 to the other; a live router on these prompts
evidently spreads a 6-row step differently. So the honest statement of the win is the
real-routing one, and the fixed-routing number is a kernel-chain bound.

### Paired per-row, the same 8 rows in every arm

The client ran the identical rows in the identical order, so per-row pairs are the
comparison least exposed to the drift noted below:

| Row isl | baseline | `attn_core_lean` | stacked | lean - base | stacked - lean |
| --- | --- | --- | --- | --- | --- |
| 30355 | 1196.8 | 1239.8 | 1227.4 | +43.0 | -12.4 |
| 30798 | 1236.0 | 1211.5 | 1215.6 | -24.5 | +4.1 |
| 31140 | 1171.8 | 1242.1 | 1226.7 | +70.3 | -15.4 |
| 31664 | 1182.1 | 1235.7 | 1201.5 | +53.6 | -34.2 |
| 32041 | 1216.6 | 1228.3 | 1232.8 | +11.7 | +4.5 |
| 32047 | 1258.8 | 1274.2 | 1250.1 | +15.4 | -24.1 |
| 32833 | 1180.4 | 1233.9 | 1217.3 | +53.5 | -16.6 |
| 34109 | 1180.2 | 1231.6 | 1284.7 | +51.4 | +53.1 |
| **median** | 1189.5 | 1234.8 | 1227.0 | **+47.2** | **-13.9** |

`attn_core_lean`: 7 of 8 rows faster, median +47.2 tok/s/user (+4.0 %), and the tail
improves more than the median (p99 step 4.121 -> 3.753 ms, -9 %).

**Stacking the two bitwise slots is not a win.** 5 of 8 rows drop, median -13.9. Both
slots engaged (they are in the served list and `attn_wo_a_fp8` logged no decline, so its
per-layer recovery was exact), so this is the pair's cost, not a refused slot. It is also
the result the -13.1 and -7.7 us in the model page should have predicted: at this noise
level the pair is not resolvable, and here it reads slightly negative. Keep `attn_core_lean`
alone at TP8 unless this pair is re-measured to a resolved gain.

## What the drift means for the method

Two things this session showed, both worth building into the next measurement:

- **Arms must be interleaved, not sequential.** Three arms ran back to back over 16
  minutes, and the last stretch of every run reads high: binned into eighths, `base`
  holds 2758-2793 us for six bins then reads 2870 and 2898, `lean` holds 2692-2790 then
  2790 and 2840, `stack` holds 2712-2806 then 2740 and 2853. The lift is 80-120 us and
  arrives at the same point in all three arms, so it is not thermal and not a property of
  the kernels: every GPU held its 2032 MHz maximum clock with no throttle reason through
  the runs, at 40-45 C. Treat the tail of a run as suspect, interleave the arms, and read
  paired differences rather than arm-wide medians.
- **The first pass after capture is not a measurement.** The baseline's warm pass ran at
  13.1 ms per step for its whole duration (the client's p50 for that pass: 17.6 ms) while
  its next two passes ran at 2.755 and 2.779 ms. Whatever settles after capture has to
  settle before the first timed pass; treat a slow first pass as a thrown-away pass, and
  look at the per-pass numbers rather than an arm-wide median.

## Real prompts (bench32), first look

`bench/bench32k_client.py` (outside this package) sends each row's text prompt to
`/generate` as ids and reads the decode step off the SSE chunk gaps, recording the same
fields as `static_table.py` plus the row's real ISL. A first 3-row look on `attn_core_lean`
read p10 2.716 ms against a p50 of 4.086 ms; the 8-row runs above supersede it (p10 2.755,
p50 2.842), so that spread was the 3-row sample, not the context length.

## Reproducing

```bash
# baseline: an explicit list, since moe_balanced needs balanced expert storage it
# does not have here; a default start leaves it off, listing it must build
SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(x for x in s if x!='moe_balanced'))")
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$SLOTS \
SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES=/tmp/stamps \
  # ... the TP8 + EP2 launch from docs/models/deepseek_v41.md

# the core_lean arm adds the slot and the layout it reads
SGLANG_DSV4_KV_LAYOUT=v41_new \
SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$(python -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join([x for x in s if x!='moe_balanced'] + ['attn_core_lean']))") \
  # ... the same launch
```

Take the step times from the in-graph stamps (`events-r0.bin` + `layout-r0.json`), one
verify-period median, and record the GPU compute process count per pass so a
contaminated arm is visible rather than mysterious.

## Open

- `attn_wo_a_fp8` and `attn_index_weights_bf16` stacked on `attn_core_lean` read
  slightly negative here (median -13.9 tok/s/user, 5 of 8 rows down); a resolved answer
  needs interleaved arms and more rows, not another sequential run.
- `attn_cluster_merge` re-measured in a reproducible state.
- An interleaved A/B/A harness: the drift above is the same size as the remaining
  headroom on this path, so it sets the resolution of everything measured after this.
