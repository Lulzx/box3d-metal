# M4 Pro merged narrow+solve (Phase 1 build)

## Host

- Apple M4 Pro, macOS 26.7, Xcode 26.6, `Release`, 8 CPU workers, substeps 4.
- No quiet-host protocol for this checkout run; treat absolute numbers as
  structural evidence, not the stored baseline (rerun under
  `docs/benchmarks/protocol.md` before comparing phases).

## What landed

Steady resident steps now encode narrow-phase and contact-solve kernels into
one command buffer with a single commit/wait (was: one buffer + blocking wait
per phase). Design:

- `b3Collide` predicts a stable-resident step (reusable inputs, zero
  exceptions/transitions and complete coverage last step, convex-only
  topology, no staleness/pending topology) and defers narrow encoding into an
  uncommitted buffer, skipping the CPU middle. All gate inputs are CPU-known.
- The solve phase encodes into the same buffer, waits once, validates
  zero-exceptions/zero-transitions from the narrow summary, and only then
  consumes solve results. Mispredicts run the full legacy middle/tail with
  the (valid) merged narrow outputs and re-solve; hard failures re-run legacy
  narrow for CPU fallback. Impulse authority is invalidated on every
  non-accept path; residency commits only on accept.
- GPU time of the shared buffer is split by dispatch share (4 narrow vs
  10+13*colors solve) for stage telemetry — documented approximation until
  per-stage counter timestamps land.
- Supporting refactors: narrow split into encode/consume halves (legacy path
  byte-identical behavior), `BOX3D_METAL_NO_MERGE=1` escape hatch,
  `mergedNarrowSolveAttempt/Accept/MispredictCount` profile counters,
  `MetalMergedNarrowSolveTest` (settle → teleport → re-settle with CPU twin).
- Narrow/solve stage attribution, command/buffer/dispatch counting, and every
  bypass/collision counter are exact on all paths (speculative bumps are
  snapshot-restored on recovery).

## Measurements (settle probe: 1 sphere on floor, recycle 0, 200 steps)

Wait-trace, steady state, same host back-to-back:

```
merged ON : narrow_merged gpu=0.054ms + solve_merged wait=0.479ms (gpu=0.310ms)
            1 buffer, 1 bubble (~0.12ms), cmd=1
merged OFF: narrow wait~0.16ms (gpu~0.03ms) + solve wait~0.33ms (gpu~0.21ms)
            2 buffers, 2 bubbles (~0.25ms), cmd=2
```

The merge removes exactly one ~0.13 ms scheduling bubble per step, matching
the spike's per-buffer measurement. Wall-clock delta on this tiny world is
within run variance (GPU execution dominates); the structural win (cmd 2→1,
one bubble) is exact.

```
200 steps: attempts=131 accepts=130 mispredicts=1, max CPU divergence 1.1e-6
```

The single mispredict is the teleport step; recovery is exact (zero
divergence across it). Full `box3d_test` green on float, double, CPU-only,
and ASan; `MetalMergedNarrowSolveTest` locks 52/51/1 plus single-buffer
steady steps. Default scene-harness worlds never engage the merge (recycle
distance defaults to nonzero, so no stable-resident steps): attempts 0
everywhere, zero behavior change.

## Findings for later phases

1. The merge's payoff is bounded by the bubble (~0.13 ms/buffer), as the
   spike predicted. The residency bypass machinery had already removed the
   CPU middle on steady steps; there was no further CPU work to overlap.
2. Residency on-ramp quirk (pre-existing, out of scope): a CPU input pack
   written for a brand-new contact snapshots bypass-ineligible state, and
   with static topology no repack ever fixes it — that contact stays a
   CPU exception forever (observed: near-rest starts). Bootstrap-authored
   inputs do not have this problem. Worth a follow-up; the merge test
   deliberately uses a drop start.
3. Next: the remaining per-step buffers (pairs when moving, readback blits)
   are the next merge candidates toward one command buffer per step.
