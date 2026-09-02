# M4 Pro body-finalization traversal bypass — 2026-09-03

## Scope

This checkpoint removes Erin's complete per-awake-body CPU finalization walk
from the bounded unconstrained Metal route. Sleeping and CCD must be disabled,
all solver constraints must be absent, and Metal must have produced complete
body state, move events, shape bounds, and resident tree refit outputs.

The device retains the advancing finalization center, including VF64 binary64
center bits in double builds. It also publishes absolute transforms and sleep
metrics, updates inverse world inertia, clears force and torque, and resets
transient solver state. The CPU `b3BodySim` array becomes a lazy mirror.
Queries, property mutation, awake-set topology changes, recording, Metal
disable, or expansion into contacts, sensors, joints, sleep, or CCD synchronize
that mirror once before CPU code consumes it.

The embedded IEEE64 Metal arithmetic is the upstream `f64-metal`/`VF64-metal`
`IEEE/Arithmetic.metal` source at commit
`729021777455da72db8809d9ef1269c677d88b3f`, with only local SPDX and provenance
lines added.

## Correctness and boundary evidence

The float and double/VF64 Metal suites cover:

- ten consecutive resident unconstrained steps against the CPU oracle;
- ten complete body-walk bypasses, with no body-sim synchronization until a
  public transform observation;
- a post-step `userData` mutation that synchronizes before invalidating and
  repacking the resident finalization-property generation;
- an unsupported revolute-joint transition that synchronizes the body-state
  and body-sim mirrors exactly once before the established CPU fallback;
- VF64 far-world transforms and conservative directed AABB narrowing.

Validation passed the float and double/VF64 warning-as-error Metal suites, full
Metal Release, full CPU Release, full Metal AddressSanitizer, full Metal
UndefinedBehaviorSanitizer, and focused double/VF64 UndefinedBehaviorSanitizer
configurations.

## Whole-world signal

Hardware was an Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB unified memory).
The Release harness used 524,288 dynamic bodies, one sphere per body, four
substeps, eight Box3D workers, Metal finalization and broad phase enabled, five
warm-up steps, and six measured whole-world steps per process.

```text
run                       1          2
CPU ms               42.330     43.800
Metal ms             22.555     22.218
Metal kernel ms      15.012     14.011
speedup                1.877x     1.971x

finalization readback bytes       0
body-state readback bytes         0
body-sim synchronizations         0
move-event readback bytes         0
body traversal bypasses          11 / 11
body-state uploads / reuses       1 / 10
body-property uploads / reuses    1 / 10
```

These are loaded-host measurements, not an accepted quiet-host result: Chrome,
WindowServer, and an unrelated `ffmpeg` process were active during capture.
They are nevertheless a repeatable performance signal and structural proof
that the former shared finalization result stream, solved-state readback, and
per-body CPU finalization traversal were absent from every warm and measured
Metal step. A quiet-host matrix remains the performance gate.
