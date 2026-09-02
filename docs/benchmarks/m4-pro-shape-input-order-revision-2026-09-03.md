# M4 Pro shape-input order revision — 2026-09-03

## Scope

Stable shape finalization previously validated its cached `bodyIndex` mapping
by scanning every awake `b3BodySim` and comparing body IDs. This checkpoint
replaces that O(body count) validation with a monotonic awake-body order
revision stored beside the persistent shape-input registry.

Create/destroy, wake/sleep, awake-set transfers, and snapshot restoration
advance the revision. A stable non-CCD step checks the cached body count and
revision in constant time, then reuses the 72-byte-per-shape input registry
without touching the CPU body array. CCD retains its separate fast-body scan.
Cold or invalidated routes still rebuild from Erin's CPU data as the oracle.

## Correctness evidence

The registry differential performs a sleep followed immediately by a wake
before the next step. The awake count returns to 64, while swap-removal plus
append changes body ordering. The revision changes twice, rejects the stale
mapping, rebuilds once, and then resumes reuse. A filter-only mutation also
forces the established rebuild path.

Both float and double/VF64 warning-as-error `MetalTest` suites pass. The
524,288-body benchmark additionally fails if its expected one pack, ten reuses,
and eleven order-revision checks are not observed.

## Whole-world signal

The Release harness used an Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB
unified memory), 524,288 dynamic bodies with one sphere each, four substeps,
eight workers, five warm-up steps, and six measured steps.

```text
run                              1          2
CPU ms                      46.210     45.750
Metal ms                    20.349     21.466
Metal kernel ms             13.656     14.333
speedup                       2.271x     2.131x

shape-input packs / reuses          1 / 10
order-revision checks              11
body traversal bypasses            11 / 11
body-sim synchronizations           0
body-state readback bytes           0
finalization readback bytes         0
move-event readback bytes           0
```

This is loaded-host directional evidence, not an accepted quiet-host result.
Chrome, WindowServer, Telegram, Playwright, and an unrelated `ffmpeg` process
were active during capture. Compared with the immediately preceding loaded-host
checkpoint (1.88–1.97x), the result is consistent with removing the final
steady CPU awake-body scan, but host load prevents attributing the entire
difference to this change. A quiet-host matrix remains required.
