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

## `attn_core_lean`: measured

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

## Real prompts (bench32)

`bench/bench32k_client.py` (outside this package) sends each row's text prompt to
`/generate` as ids and reads the decode step off the SSE chunk gaps, recording the same
fields as `static_table.py` plus the row's real ISL. Preliminary, on `attn_core_lean`,
3 rows of `lca-project-level-code-completion__small_context`, `--osl 256`:

| isl | tok/s/user |
| --- | --- |
| 32041 | 1276.3 |
| 27588 | 767.5 |
| 32314 | 764.1 |

Summary: step p50 4.086 ms, p10 2.716 ms, p90 5.527 ms, TTFT p50 454.8 ms.

Two things this already says. The p10 (2.716 ms) sits on the fixed-routing p50
(2.722 ms), so the fastest real-routing steps hit the same floor as the synthetic
measurement; the p50 is 1.4 ms higher because real routing spreads each 6-row step over
more distinct experts and the 32k context makes the compressed-index side heavier. And 3
rows is too few to report: the first row is 66 % faster than the other two. A baseline
arm on the same rows is still needed before any real-prompt delta is claimed.

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

- `attn_wo_a_fp8` and `attn_index_weights_bf16` stacked on `attn_core_lean`, same
  method: `attn_wo_a_fp8` streams the very `wo_a` the lean core feeds, so the pair is
  worth measuring rather than assuming.
- `attn_cluster_merge` re-measured in a reproducible state.
- A bench32 baseline arm, and enough rows that the per-row spread is bracketed.
