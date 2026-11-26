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

The block size and the acceptance length are different quantities and the two get
confused. Block size 5 is how many draft tokens a step *proposes*; the verify half is
always 6 rows (1 anchor + 5 draft) whatever happens. The acceptance length is how many of
those 5 the verify *keeps*, averaged over steps, so a step commits `1 + accepted` tokens.
3.51 is not a design constant: it is the mean measured on an earlier TP4 run under forced
acceptance, adopted here so arms share one verify width. With acceptance `a` a step's
time is set by the 6-row verify and barely moves, while throughput and TPOT scale with
`a`: TPOT = step time / `a`. Two earlier bench32 runs read mean acceptances of 4.50 and
4.35 (backed out as tok/s x step time); both ran `ep_size=1` on the mirage static hook
rather than this package's megakernel, so they bound the arithmetic, not this stack's
tok/s. At the same step time a run forced to 3.51 reports about 3.51/4.50 = 78 % of what
those read. Read every tok/s and TPOT number below against the acceptance it was taken
at.
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

## The acceptance length, measured

The numbers in this document hold acceptance at 3.51. That sits below what these prompts
actually reach, so the tok/s figures here are deliberately low. Two lean launches on the
same stack, same slots and layout, differing only in `SGLANG_SIMULATE_ACC_LEN`, over the
same 8 bench32 rows:

| Forced acceptance | Period median | bench32 step p50 | bench32 tok/s/user p50 | TPOT |
| --- | --- | --- | --- | --- |
| 3.51 | 2725.9 us | 2.828 ms | 1233.2 | 806 us |
| 5.10 | 2748.9 us | 2.820 ms | 1814.4 | 553 us |

The step time does not move (+23 us in the stamps, -8 us in the client, both inside the
noise) while throughput rises 47 %. So on this stack the acceptance length sets tok/s and
TPOT alone, at `TPOT = step / acceptance`, and a step time measured at one acceptance
carries over to another.

That is the whole of the difference between the 1500 tok/s an earlier bench32 run
reported and the 1233 here: the earlier run sat at a mean acceptance of 4.50 and this one
was forced to 3.51 (3.51/4.50 = 78 %, and 1233/1503 = 82 %). It was not a regression, and
the earlier run is not a baseline for this one in any case: it ran `ep_size=1` and the
mirage static hook, not this package's megakernel.

Report tok/s and TPOT together with the acceptance they were taken at; without it the
numbers are not comparable across runs.

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

## Slots that do not build at TP8 + EP2

Worth recording so the next attempt does not repeat it. `draft_final_collapse` needs
`draft_hop`, and `draft_stage0_first` needs `draft_stage0_table`. At attention TP8 + EP2
the draft runs its own three-stage set (`draft_attn`, `draft_qfront`, `draft_wo_a`,
`draft_swapab`, `draft_qkv_a`, `draft_wo_b_ll`, ...) and neither `draft_hop` nor
`draft_stage0_table` is among the served slots, so listing the two dependents builds
nothing. They were listed and simply did not appear in the served list; no error, no log
line. Check the served list after a launch rather than assuming a listed slot took.

`attn_index_q_first` did build and serve, and reads +17.9 tok/s on one run (1788.3 ->
1806.2, step p50 2.843 -> 2.789 ms). One run, and this node's launch-to-launch spread is
1.4 %, so it is not confirmed either; it is a candidate for the interleaved harness below,
not a win to report.

## What to do next

The MoE is where the remaining time is. Each verify step spends about 28 us a layer on the
routed pair (45 % of a layer's 61.6 us) and reads roughly 44 MB per rank to do it: 6 rows,
~19 distinct experts, ~10 of them per EP half, ~4.6 MB an expert per MoE-TP4 rank
(`GEMM1_UNITS` 9 x 128 rows of w13 over K 5120 at MXFP4, plus 5120 x 640 of w2).

That 28 us is an in-model attribution, not the kernel's own cost. Measured in isolation
with `benchmark/bench_deepseek_v41_routed_moe.py` (one EP2 rank's window,
`--local-experts 192 --offset 0`, 3 repeats, spread <= 0.03 us, spacer of two empty
kernels in the same graph subtracted), the shipped pair is **15.15 us a half** at 19
distinct experts (19.14 us with a shared-expert-sized chain beside it on a second stream).

The shape of that number is the point, and it is not what the previous version of this
section claimed. Sweeping the distinct-expert count:

| distinct | `pipe_clc` us/half |
|---------:|-------------------:|
| 1        | 11.41              |
| 4        | 11.81              |
| 8        | 12.28              |
| 19       | 15.15              |

An extra distinct expert costs 0.21 us, i.e. ~0.42 us for the ~half of them this rank
owns, which for ~4.6 MB is faster than the part's memory could deliver it. Adding work is
therefore nearly free, and **the half is fixed-latency bound, not bandwidth bound**: at 19
experts the whole half averages 2.9 TB/s but 11.4 of its 15.2 us is the floor at one
expert, where only ~2.3 MB moves and the roofline would be 0.3 us. For scale, the same
bench's `--bandwidth` probe reads a 4 GiB buffer at **6.29 TB/s** on this part, so the
routed pair at the live routing point runs at 47 % of what the memory can actually
deliver (not 20 % against a spec 8 TB/s). Neither occupancy nor the row count is the
limiter, so widening the tiles or splitting K buys nothing.

### Tried and reverted: the two cheap overlap levers

Both were measured in `bench_deepseek_v41_routed_moe.py` and both are back out of the
tree; do not redo them.

- Releasing the dependents at CTA entry (an extra `griddepcontrol.launch_dependents` in
  `gemm1.cu` and `gemm2_pipe.cu`, leaving the existing end-of-loop release to no-op)
  costs rather than saves: 15.15 -> 15.65 us at D=19 and 19.14 -> 22.67 us with the side
  stream, because the next stage's CTAs go resident early, pay their mbarrier init and
  squat on SMs the current stage still needs. Correctness is unaffected either way (a
  dependent reads only after its own wait).
- Widening the K-loop throttle from 3 to 5 stages (5 barriers each, `throttle_empty`
  moved to 240 so the wider `throttle_full` fits) is neutral at the live routing point,
  15.04 -> 15.04 us, and worse at one expert, 10.88 -> 12.78 us. The 5-stage, 160 KB
  A buffer was therefore not held back by that throttle.

### The trap: a shorter grid looks 25 % faster and is 25 % of the work

`benchmark/bench_deepseek_v41_routed_moe.py`'s own grid is `(units, ctas)` = 1440 blocks
at the live routing point, which invites "each block does one tile, so shrink the grid and
make the CTAs persistent". Sweeping the y extent on the shipped `pipe_clc` build reads:

| grid y | side=0 us | side=1 us |
|-------:|----------:|----------:|
| 36 (shipped) | 15.04 | 19.08 |
| 8  | 13.93 | 16.77 |
| 4  | 12.52 | 14.89 |
| 2  | 11.46 | 12.59 |
| 1  | 11.04 | 12.32 |

All of it is dropped work. Comparing the outputs of a one-wave grid against the shipped
one, `Y` is zero in 112640 of 122880 elements (92 %) of the first half. The kernel's work
queue is cluster launch control, which hands out only blocks that have not launched yet:
a grid that already covers the tile space gets nothing back, so shortening it throws tiles
away instead of concentrating them.

And persistence is already in place, so there is nothing to add. Measured with ncu on the
shipped grid at D=19:

```
launch__grid_size                  1440
sm__ctas_launched.sum               148     one CTA a multiprocessor
launch__waves_per_multiprocessor   9.73     each resident CTA runs 9.73 tiles
launch__occupancy_limit_shared_mem  1 block
```

Only 148 of the 1440 blocks ever launch: each resident CTA pulls its next tile through
CLC cancellation, 9.73 tiles apiece, exactly grid / SMs. So the fill is already amortised
over ~10 tiles and no grid or scheduling change can reduce it further. A hand-written
strided tile walk (replacing the CLC answer in each of the five role warps) was tried as
well: at the shipped grid it desynchronises the roles and produced NaNs, and it has no win
to recover even when correct, because the thing it adds already happens.

What is left is the fill itself, paid once a launch: 2 kernels x 39 halves = 78 fills a
verify step. That is the 42 % intra-CTA barrier stall ncu reports, and reducing it means
restructuring the prologue of generated code, not a knob. Merging GEMM1 and GEMM2 into one
launch would halve the fills, but it needs a grid-wide barrier between them, which is only
deadlock-free if every CTA is resident (grid <= 148); GEMM1's grid is 9 x 36 = 324, and
shrinking it is exactly the trap above, so a fusion also has to solve the tile walk first.

### The side stream's +4.0 us is mostly the shared expert itself

Having the shared-expert-sized chain beside the routed pair costs 19.08 vs 15.04 us. It is
not wasted scheduling. Run alone (routed pair removed), that chain takes 10.99 us a half
including the 1.26 us spacer, so ~9.7 us of real work: separately 15.04 + 9.7 = 24.7 us,
together 19.08, i.e. the overlap already recovers 5.6 us and only 4.0 us stays exposed.
The exposed part is not reachable by reserving SMs either: the routed pair occupies all
148 multiprocessors (180 KB of smem a CTA, `launch__occupancy_limit_shared_mem` 1), and a
grid cut below the resident capacity has every block launch, so CLC has nothing left to
hand out and the leftover tiles are dropped -- the same trap as above, from the other side.

### GEMM1 is the badly amortised one, and its fix is out of scope

Profiled on the shipped grid at D=19 (`pipe_clc`, side 0):

| | GEMM1 | GEMM2 |
|---|---:|---:|
| grid | 324 (9 x 36) | 1440 (40 x 36) |
| CTAs launched | 148 | 148 |
| waves per SM | **2.19** | 9.73 |
| duration | 13.44 us | 9.95 us |
| DRAM | 33.6 % | 25.3 % |
| SM | 17.8 % | 14.0 % |
| active warps | 24.8 % | ~25 % |
| occupancy limit (smem) | 1 block | 1 block |
| registers/thread | 128 | ~96 (the reg pool) |

GEMM1 is the worse of the two: its tile space is only 9 x 36, so the fill is amortised
over 2.19 tiles against GEMM2's 9.73, and that is why a kernel that moves more bytes than
GEMM2 (1152 x 5120 a expert against 5120 x 640) runs 3.5 us longer at a *higher* DRAM
percentage.

Both ways to fix it are generator decisions, not edits to the sources under
`sm103/moe/src`:

- Two CTAs a multiprocessor would hide the fill behind a co-resident CTA, but it needs the
  smem below 113 KB (GEMM1 asks 216832), the `__launch_bounds__(512, 1)` second argument
  raised, and the register cap halved from 128 -- i.e. a different stage depth in
  `launch.json` and the register pool, not a source change.
- More tiles per CTA would need smaller M tiles, and tcgen05 `cta_group::1` has M = 128,
  which is what makes 1152 rows exactly 9 units.

So the fill stays until the routed pair is rebuilt at a profile that fits two CTAs an SM;
that lives in the generator, outside `sgl-nvidia-mega`.

### The CTA prologue is only 0.83 us, and a deeper K pipeline is the wrong direction

Instrumented `gemm1.cu` with `%globaltimer` stamps through the clamp pointer (read-only in
the work loop) and read them back on one CTA: entry -> barrier init 0.32 us, -> tmem alloc
0.19 us, -> kernel exit 0.32 us, **0.83 us of prologue in total**. Optimising the prologue
is therefore not a lever, and a 7.45 us figure that a zero-tile run produces is an
artefact: with no tiles to steal, cluster launch control cancels nothing, all 1440 blocks
launch one after another (9.73 serial waves at 1 CTA an SM) and each pays its 0.83 us.
Measured that way a zero-tile run is 6.09 us at grid y = 1 and 7.45 us at y = 36, of which
the grid-dependent part (the drain) is only 1.36 us.

GEMM1's K loop is 10 chunks against 3 stages, which looks like an under-filled pipeline and
invites "use more stages". That is backwards here: fill costs one stage each, and each CTA
only runs **2.19 tiles** (GEMM2 runs 9.73), so this launch is ~45 % fill already and every
extra stage adds more prologue than it hides. The lever would be fewer, larger chunks, and
that needs the A box in `_weight_map` and the four MMA descriptor offsets in `gemm1.cu`
(+0, +16384, +32768, +49152 within a 65536-byte stage) to change together -- generator
tiling, not a local edit.

The measured shape of the routed pair, all in one place: marginal expert 0.21 us, first
expert 11.4 us (D=1), 19 experts 15.04 us; GEMM1 alone 13.44 us, GEMM2 alone 9.95 us, the
pair 15.04 us, so PDL already overlaps most of GEMM1 under GEMM2. It is a latency chain,
not a bandwidth or occupancy problem, and the chain is the K loop.

Cross-process variance, distinct from the 0.03 us spread inside a run: the D=1 floor
rebuilt and re-run came back 11.41 and then 10.88 us, so compare across launches only
with ~0.5 us of slack.

What does have headroom, in the order the measurements support it:

- The fill itself: ~5-9 us a launch, 78 launches a verify step. This is the per-CTA
  prologue (mbarrier init, tensormap first use, tmem alloc) counted above, and it is the
  only remaining MoE lever. It needs the generated prologue restructured, or a fused
  GEMM1+GEMM2 launch (one fill a half instead of two) that also solves the tile walk so a
  grid-wide barrier stays deadlock-free.
- The side stream: putting the shared-expert chain beside the pair costs a flat +4.0 us
  (19.14 vs 15.15) under every variant, ~8 % of a verify step over 39 halves, and it is a
  scheduling change rather than kernel work.
- The variant is already the best one: `pipe_clc` (slot `gemm2_pipe` + `gemm2_reg_pool`,
  both default-on) beats the plain build by 0.6 us a half at every D, and `pool_clc` ties
  it. Taking the route kernel off the chain (`preroute+`) is 0.4 us *slower*, so routing is
  already fully hidden and is not a candidate.

Before that, the harness: with the remaining per-slot effects at 0.5-1.5 % and this
node's launch-to-launch spread at 1.4 %, a sequential A/B cannot resolve them. Interleave
the arms inside one launch (same server, alternating timed windows) and read paired
differences.

## Open

- `attn_wo_a_fp8` and `attn_index_weights_bf16` stacked on `attn_core_lean` read
  slightly negative here (median -13.9 tok/s/user, 5 of 8 rows down); a resolved answer
  needs interleaved arms and more rows, not another sequential run.
- `attn_cluster_merge` re-measured in a reproducible state.
- An interleaved A/B/A harness: the drift above is the same size as the remaining
  headroom on this path, so it sets the resolution of everything measured after this.

### Engram cannot live on the GPU under the megakernel, and the end-to-end nsys capture cannot see the decode step

Three end-to-end results on the TP8 + EP2 target (8x B300, DSpark block 5, bench32 real
prompts, `--enable-nvidia-megakernel`).

**Engram placement is a hard constraint, not a tuning knob.** Launching with
`--engram-config '{"placement": "gpu"}'` is rejected at startup:
`--enable-nvidia-megakernel cannot run this setup: engram_config placement 'gpu',
replication 'none': the Engram prefetch reads the per-rank host tables`. The megakernel's
Engram prefetch is built on the per-rank host table, so `cpu` is the only legal placement
while the megakernel is on. There is no CPU-vs-GPU Engram comparison to run.

**The CPU arm is the reference: 1452.4 tok/s/user, step p50 3.532 ms** (step p99 4.81 ms,
TTFT 407 ms) over 8 rows, and 1426.2 tok/s / 3.559 ms with nsys attached (~1.8 % tracer
overhead). So the tracer costs less than the run-to-run spread and can be left on.

**No megakernel kernel appears in the trace.** `report1.nsys-rep` (33.9 MB) holds 302 424
kernels over a 4.60 s span and 195 distinct names, all flashinfer / cutlass / nvjet /
torch / cub / sglang / deep_gemm -- not one of the plugin's kernels. Two causes:

- Eight ranks in one session overrun the default CUDA buffer; only ~4.6 s of a ~11 s pass
  survived, so the decode windows are mostly outside the capture.
- Decode runs under a CUDA graph and nsys defaults to `--cuda-graph-trace=graph`, which
  records the graph as one node. The per-step kernels are not expanded.

What the captured slice does say, and it is worth recording: 39.9 % of kernel time is
`trtllm_mnnvl_allreduce::twoshotAllreduceKernel` at 719 us a call, and 16.0 % is the
prefill sparse attention -- i.e. **the captured region is prefill, and prefill is
communication-bound**, which says nothing about the B=1 decode target. Engram itself is
cheap there: `_engram_gather_kernel` 355 us x 432 and `_engram_gate_kernel` 189 us x 432,
both prefill-scale, and 41 us a call in the isolated decode window. Engram on CPU is not a
decode bottleneck.

A usable decode profile needs `--cuda-graph-trace=node` with a raised
`--cuda-buffer-size`, and a decode-only window (short ISL, or a cached prefix) so ~100
steps fit in the buffer.

## Open (end to end)

- Re-capture decode with `--cuda-graph-trace=node`; every conclusion about which part of
  the 3.53 ms step dominates is currently unsupported.
- `--enable-layerwise-nvtx-marker` would give per-layer ranges directly and is the cheaper
  route to the same answer.

### The decode profile, and the two things it says: 23 % of the step is dead GPU time, and the mHC coefficients are the largest kernel

Captured decode on real 32k bench32k prompts with `--cuda-graph-trace=node` and the
capture window opened only around the measured request. The warmup row fills the radix
cache, so the measured row's prefill is ~0 (TTFT 135 ms after a 27 s warmup) and the
window is decode. 50 complete steps, each bounded by the plugin's own `dsv41_bench_stamp`.

**Fast path now on by default.** `attn_core_lean` moved from `OPT_IN_SLOTS` into
`SHIPPED_SLOTS`/`DEFAULT_ON_SLOTS`, gated by `runtime._core_lean_on()`: it runs only when
the v41_new KV layout is selected and `attn`, `swapab_h8`, `wo_a_g1` are all present, and
turns itself off (rather than rejecting the request) when any is missing. The build's slot
line now reads `muLLM's one-launch core 1-8 (v41_new KV layout only)` on every layer
group. In the trace the swap-AB sparse MLA kernel (`flash_fwd_splitkv_mla_fp8_sparse`, was
261 us/step) is replaced by `dsv41_core_lean_sparse_early` at 237 us/step.

Per step (mean of 50, device 0, all eight ranks run the same shape):

| | us/step | share |
|---|---|---|
| step wall (stamp to stamp) | 3606 | |
| **union GPU busy** | **2781** | **77.1 %** |
| **apparent GPU idle** | **825** | **22.9 %** |
| kernel work (sum of durations) | 6153 | 2.21x parallelism |

**Most of that "idle" is the tracer, not the machine.** The same process measured the same
row twice: once with the capture window open and once with `nsys` running but not
collecting, so the only difference is the collection itself.

| run | step p50 |
|---|---|
| no `nsys` at all (`ab_nomoe`, 8 bench32 rows) | 2.835 ms |
| `nsys` attached, **not** collecting (`n32k3-warm`) | **2.798 ms** |
| `nsys` **collecting**, `--cuda-graph-trace=node` (`n32k3-dec`) | 3.432 ms |

Node-mode collection costs ~0.60 ms a step, against a 0.57 ms warm/untraced delta of 0.04.
So of the 825 us of apparent idle, ~600 us is the tracer re-instrumenting every node of the
graph and the real idle is roughly **225 us, about 8 % of the step** -- a different problem
from the 23 % it looks like. Read the traced 3606 us as an upper bound, not as the shipped
step.

What is real behind the number: per rank the host does ~115 CUDA API calls a step and spends
~788 us in them (22 % of the step, itself tracer-inflated), split by gap into 516 us of
host-API stalls over 4.4 gaps and 193 us of pure-Python stalls over 1.1 gaps. 98.2 % of the
48233 kernels in the window are inside the CUDA graph; the 17.5 a step that are not are the
`at::native::` elementwise / `multi_tensor_apply` / `index_elementwise` tail, and those are
exactly the gap points (`cudaLaunchKernel_v7000` 16.4 a step, each paired 1:1 with a
`cuKernelGetName`, plus a `cudaStreamIsCapturing` 24 a step). So the lever is fewer host
launches a step -- graph-capture or drop that tail -- and it is worth ~0.2 ms, not 0.8.

Kernel work, in order (`us/step` is the sum of durations, so these overlap 2.21x):

| us/step | n/step | kernel |
|---|---|---|
| 1070.6 | 86.0 | `dsv41_hc_coeff` |
| 576.6 | 80.0 | `all_reduce_mhc_LamportMHC*` |
| 499.4 | 43.0 | `dsv41_routed_expert_gemm1` |
| 439.5 | 43.0 | `SwigluLowLatency` |
| 371.6 | 58.0 | `dsl low_latency_blockscaled` |
   550|| 339.2 | 43.0 | `dsv41_wo_b_split` |
| 330.8 | 43.0 | `dsv41_wo_a_fused` |
| 320.4 | 43.0 | `dsv41_routed_expert_gemm2` |
| 237.0 | 38.0 | `dsv41_core_lean_sparse_early` |
| 232.1 | 32.0 | `dsv41_q_cluster8_fit` |

**Read that table as work, not as cost.** It is the sum of each kernel's own duration, and
these kernels run 2.21x in parallel, so a kernel being top of this table says nothing about
what a second of it buys. The question that decides where to optimise is *how much the step
shrinks if the kernel is not there*, i.e. `union(busy) - union(busy without k)`:

| crit us/step | work us/step | n/step | kernel |
|---|---|---|---|
| **263** | 577 | 80 | `all_reduce_mhc_LamportMHC*` |
| **180** | 320 | 43 | `dsv41_routed_expert_gemm2` |
| **60** | 92 | 40 | `all_reduce_mhc` (draft side) |
| **49** | 116 | 43 | `dsv41_router_mma` |
| 21 | 1071 | 86 | `dsv41_hc_coeff` |
| 11 | 237 | 38 | `dsv41_core_lean_sparse_early` |
| 8 | 339 | 43 | `dsv41_wo_b_split` |
| 7 | 499 | 43 | `dsv41_routed_expert_gemm1` |
| 6 | 279 | 46 | `dense_blockscaled` |
| 5 | 232 | 32 | `dsv41_q_cluster8_fit` |
| 1 | 372 | 58 | `dsl low_latency_blockscaled` |
| 0 | 440 | 43 | `SwigluLowLatency` |
| 0 | 331 | 43 | `dsv41_wo_a_fused` |

`dsv41_hc_coeff` is the largest consumer of GPU *work* and is **almost entirely hidden**:
removing it entirely would shorten a 2835 us step by **21 us, 0.7 %**. It overlaps with the
rest of the layer chain and the GPU stays busy. Anybody who tunes it to raise its occupancy
is tuning a kernel that costs nothing. (This corrects an earlier reading of the same trace
that mistook the work column for the cost column.)

**The critical path is communication.** `all_reduce_mhc` 263 + a second `all_reduce_mhc` 60
= 323 of the ~611 us of critical-path kernel time, then `routed_expert_gemm2` 180 (the EP2
combine) and `router_mma` 49. At batch 1 with TP8 + EP2 the payloads are tiny, so 80
all-reduces a step at ~3.3 us of exposed latency each are pure latency, not bandwidth.

For completeness, the step is not idle-bound either: SM-slot occupancy summed over the
window is 176 % of 148 SMs x the step wall, so the GPU is oversubscribed, not starved. The
levers left are the dependency chain (all-reduce latency, the EP2 combine) and removing
work from it, not finding idle time.

**Why `dsv41_hc_coeff` looks expensive and is not.** 86 launches a step (43 layers, draft
and verify) at 12.5 us each, on a grid of (`slices`, 1, 1) = **16 CTAs** -- `SLICES = 16` is
`prenorm.SLICES`, DeepGEMM's `kNumSplits`, which the kernel's slice-ordered tail reduce must
keep to stay bit for bit. K is `hc * hidden` = 4 * 5120 = 20480, so 320 K-blocks of 64 land
20 a split with no remainder and each CTA walks all 20. Profiled at the decode shape
(`rows = 1`) with `ncu`: duration 12.48 us, but **SM Active Cycles 1276 of 25241 elapsed
(5 %)**, 0.11 waves, DRAM 2.15 %, compute 1.39 %, achieved occupancy 12 %. So each launch is
almost all fixed cost and the SMs are idle for most of it -- which is exactly why it hides
behind the rest of the chain instead of extending it.

**Two CTAs an SM is closed, two ways.** The kernel's smem is 192.4 KiB
(`smem_bytes()` = 197008: 12 stages x 16 KiB + barriers), not 92 KB, against a 227 KiB SM, so
one CTA excludes a second outright; `ncu` also shows `Block Limit Registers 1` at 157
registers a thread. And it would not help anyway: the grid is 16 CTAs in total, so two a SM
would pack them onto 8 SMs, not reach more of the 148. Reaching more SMs needs a larger
`kNumSplits`, and that changes which partial sums the tail adds, so it is a numerics change,
not a free win.

**Depth is not the answer either.** Sweeping `HC_STAGES` in a captured replay of 86
launches: 12 stages (192.4 KiB) 8.99 us, 8 (128.3 KiB) 8.86, 6 (96.2 KiB) 8.90, 4 (64.1 KiB)
8.95, 3 (48.1 KiB) 10.18, 2 (32.1 KiB) 11.79 -- flat until the pipeline is too shallow, and
all outputs bit for bit equal to 12. So the 192 KiB of smem costs nothing and 12 is already
the right depth.

## Open (decode profile)

- **The critical path is the all-reduces** (`all_reduce_mhc` 263 + 60 us, `routed_expert_gemm2`
  180, `router_mma` 49 of ~611 us). At batch 1 with TP8 + EP2 the payload is tiny, so this is
  latency, not bandwidth: 80 all-reduces a step at ~3.3 us of exposed latency each. That is
  where the next real win is.
- **The apparent 825 us of idle is mostly the tracer.** Measured control: the same row costs
  3.432 ms collected (`--cuda-graph-trace=node`) and 2.798 ms with `nsys` attached but not
  collecting, so ~600 us a step is collection. The real idle is ~225 us (8 %), and it is the
  ~17.5 host launches a step that sit outside the CUDA graph -- graph-capture that tail or
  delete it. Do not chase it with a kernel, and do not trust traced step times as shipped
  numbers.
- **`dsv41_hc_coeff` is a non-target**: 1071 us of work a step but 21 us of critical path, so
  even a perfect rewrite of it buys 0.7 %. Leave it alone unless the occupancy problem is
  interesting for its own sake.
- **SM slots are oversubscribed, not starved** (176 % of 148 x step), so "find idle time" is
  the wrong frame for this step. Remove work from the dependency chain, do not hunt for gaps.
