# M4 Pro async-stepping spike (Phase 1 measure-only)

## Host

- Apple M4 Pro, macOS 26.7, Xcode 26.6, `Release`, 8 CPU workers, substeps 4.
- No quiet-host protocol for this checkout run; treat absolute numbers as
  structural evidence (wait vs GPU split), not the stored baseline (rerun
  under `docs/benchmarks/protocol.md` before comparing phases).

## Method

New env-gated wait trace (`BOX3D_METAL_WAIT_TRACE=1`, off by default, no
behavior change): `b3MetalFillStats` logs one stderr line per command buffer
with site tag, blocking wait ms, GPU execution ms (`GPUEndTime-GPUStartTime`),
encode ms, and dispatch/buffer counts. 8 sites tagged: `integrate_positions`,
`integrate_unconstrained`, `finalize_bodies`, `pairs`, `pairs_retry`,
`pairs_copy`, `narrow_phase`, `solve`. Readback/query blits outside the step
path are untraced.

## Per-site breakdown (last steady step)

`large_pyramid`, default Metal opts (2 buffers):

```
narrow_phase wait=0.369ms gpu=0.240ms encode=0.008ms dispatches=4
solve        wait=4.212ms gpu=4.087ms encode=0.021ms dispatches=127
total wait 4.581ms; step wall metal 6.14ms vs cpu 1.44ms
```

`many_pyramids`, default Metal opts (2 buffers):

```
narrow_phase wait=1.086ms gpu=0.592ms encode=0.011ms dispatches=4
solve        wait=5.244ms gpu=5.091ms encode=0.039ms dispatches=127
total wait 6.330ms; step wall metal 7.79ms vs cpu 2.94ms
```

`large_pyramid` with `BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1`
(3 buffers):

```
pairs        wait=0.365ms gpu=0.245ms encode=0.012ms dispatches=8
narrow_phase wait=0.342ms gpu=0.213ms encode=0.006ms dispatches=4
solve        wait=5.178ms gpu=5.052ms encode=0.044ms dispatches=127
total wait 5.885ms; step wall metal 7.11ms vs cpu 1.44ms
```

Full-suite trace sanity: 286 wait lines across `box3d_test`, all sites fire.

## Findings

1. **Blocking wait is ~95% genuine GPU execution, not overhead.** Per-buffer
   bubble (wait − GPU) is a consistent ~0.13 ms regardless of buffer size
   (pairs, narrow, and solve all show it). Merging N buffers into one saves
   ~0.13 ms per eliminated buffer: ~0.26 ms for the 3→1 merge.
2. **Small-buffer anomaly:** `many_pyramids` narrow shows a 0.49 ms bubble
   (wait 1.09 vs GPU 0.59) — the only site where scheduling dominates. Merging
   narrow into the solve buffer reclaims most of it.
3. **Async overlap has a hard ceiling on these scenes.** Even perfect
   CPU/GPU overlap cannot close the gap: solve GPU execution alone
   (4.1–5.1 ms) exceeds the entire CPU wall (1.4–2.9 ms). Overlapping the
   CPU contact-creation/filtering gaps between pairs→narrow→solve hides at
   most the ~1 ms of non-GPU step time.
4. **Encode is negligible everywhere** (0.006–0.054 ms per buffer); no
   encode-side work justifies async complexity on its own.

## Recommendation for Phase 1 build

Do the cheap structural merge first (narrow+ solve, then pairs, into one
command buffer on the resident path — saves ~0.3–0.7 ms of bubble), and only
then invest in lazy-wait overlap of CPU filtering/creation with GPU
execution. Neither beats the CPU wall on these scenes until GPU solve
execution itself drops; report Phase 1 against larger scenes where solve
is bandwidth-bound, not these latency-bound ones.

## What landed

- `src/metal_backend.m`: `b3MetalFillStats` takes a `site` tag and emits the
  `b3metal_wait` trace line when `BOX3D_METAL_WAIT_TRACE=1` is set. Trace off
  by default; stats accumulation unchanged.

No behavior change: full `box3d_test` suite green; trace verified silent
without the env var.
