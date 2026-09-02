# M4 Pro lazy body-state synchronization — 2026-09-03

## Scope

This checkpoint removes the unconditional solved-state copy from the bounded
unconstrained Metal route. With sleeping and CCD disabled, the CPU finalization
task consumes the shared Metal `b3BodyState` array directly and leaves that
array authoritative across steps. A public state query or mutation, awake-set
topology change, recording, Metal shutdown, or constrained/unsupported solver
route materializes the full CPU mirror once under a synchronization lock.

Constrained Metal solves still perform the existing eager state copy because
CPU constraint preparation can consume the awake-state array before dispatch.
The CPU per-body finalization bookkeeping walk also remains; this checkpoint
removes the solved-state stream copy, not that traversal.

## Structural run

Hardware was an Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB unified memory).
The Release benchmark used 512 dynamic bodies, one sphere per body, four
substeps, eight Box3D workers, Metal finalization and broad phase enabled, five
warm-up steps, and six measured whole-world steps.

```text
body-state uploads / reuses       1 / 10
body-state revision checks        11
body-state lazy syncs             0
latest state upload bytes         0
latest state readback bytes       0
body-property uploads / reuses    1 / 10
latest property upload bytes      0
finalization readback bytes       0
move-event readback bytes         0
pair fallbacks                    0
```

The observed whole-world timing was 0.174 ms CPU versus 0.499 ms Metal
(0.348x). This short local sample is structural evidence only, not a stable
performance claim. It does show that the measured steady steps no longer copy
the 28,672-byte (`512 * sizeof(b3BodyState)`) solved-state array.

## Boundary evidence

The float and double/VF64 Metal differential suite verifies both directions:

- A public velocity query after an unconstrained resident step performs one
  full lazy sync, reports `bodyStateSyncCount == 1`, and preserves the CPU
  oracle within the existing tolerances.
- An unconstrained resident step followed by an unsupported revolute-joint
  route performs exactly one state sync before CPU preparation, then takes the
  established deterministic fallback path.
- A ten-step constrained convex-contact world retains eager state readback and
  reports zero lazy syncs, making the still-CPU-dependent boundary explicit.

Validation completed with the float warning-as-error Metal suite, the
double/VF64 warning-as-error Metal suite, full Metal Release, full CPU Release,
full Metal AddressSanitizer, full Metal UndefinedBehaviorSanitizer, and focused
double/VF64 UndefinedBehaviorSanitizer configurations.
