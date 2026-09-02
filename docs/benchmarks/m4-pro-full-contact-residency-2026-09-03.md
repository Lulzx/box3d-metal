# M4 Pro fully resident convex contacts — 2026-09-03

## Scope

This checkpoint extends device authority from unconstrained worlds to stable,
zero-exception convex-contact worlds. The admitted route requires sleeping and
CCD disabled, the Metal broad phase enabled, a revision-current contact-input
registry, complete resident sphere-sphere, capsule-sphere, capsule-capsule, or
compact hull-sphere coverage, GPU contact preparation, and no mesh, overflow,
joint, callback, hit-event, or topology exception.

After the cold first-touch step, the same device-resident chain now covers
narrow phase, contact preparation and solve, final body transforms and state,
shape AABBs, tree refit, and the next collision step. It omits:

- the solved-state copy into the CPU awake-state array;
- the complete per-awake-body CPU finalization walk;
- the full shape-result blit and flat CPU apply traversal;
- pre-collision body-sim and shape-bound synchronization.

If contact inputs need rebuilding, Metal emits a compact CPU exception, or the
route becomes unsupported, the collision boundary synchronizes body sims and
shape bounds before Erin's CPU collision task can consume them. The solver
continues to materialize state before a CPU-only constraint route.

## Differential evidence

The 81-contact CPU/Metal differential covers 64 sphere pairs and 17 capsule
pairs, including SIMD tails, two-point manifolds, friction, rolling resistance,
tangent velocity, restitution, and warm starting. Its first-touch step remains
the CPU topology seed. The following three steps report:

```text
body finalization walks bypassed     3 / 3
full shape-result applies            0 / 3 (1 cold apply total)
body-state synchronizations          0
latest body-state readback bytes     0
body-sim synchronizations            0
CPU collision exceptions             0
max transform error                  2.69e-05
max velocity error                   3.58e-07
```

One public shape-AABB query also stages exactly one 64-byte result before the
later route-change bulk synchronization. The existing hit-event, input-registry
mutation, pre-solve callback, collision fallback, unsupported-joint, sleep, and
Metal-disable fixtures retain their CPU exception/materialization behavior.

Validation passed the float and double/VF64 warning-as-error Metal suites, full
Metal Release, full CPU Release, full Metal AddressSanitizer with macOS leak
detection disabled, full Metal UndefinedBehaviorSanitizer, and focused
double/VF64 UndefinedBehaviorSanitizer configurations.

## Whole-world signal

The Release resident-contact harness now enables finalization and the Metal broad
phase. On an Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB unified memory),
65,536 independent sphere contacts, four substeps, eight workers, eight warmups,
and eight measured steps produced:

```text
run                       1          2
CPU ms                5.946      5.541
Metal ms              4.795      4.765
speedup                1.240x     1.163x

body traversal bypasses          15 / 15
shape applies                     1 cold, 0 steady
body-state readback bytes         0
body-state synchronizations       0
body-sim synchronizations         0
shape-bound synchronizations      0
latest collision exceptions       0
```

These are loaded-host measurements, not a quiet-host acceptance result.
WindowServer, Telegram, Terminal, Chrome, and other interactive processes were
active. The result is a directional end-to-end signal and structural proof that
the former state copy, body walk, and shape apply stream are absent from every
warm and measured Metal step. A quiet-host matrix remains pending.
