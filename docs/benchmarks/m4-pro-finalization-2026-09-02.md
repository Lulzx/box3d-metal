# M4 Pro experimental GPU finalization — 2026-09-02

## Scope

This ports the arithmetic portion of Box3D body finalization to Metal: final
rotation, body-origin offset, farthest-point motion and sleep metrics, and
world-space inverse inertia. Fused solver paths append it to their existing
command buffer and reuse resident body state. CPU code still owns large-world
position accumulation, CCD, events, island sleep mutation, shape AABBs, and
broad-phase updates.

The stage is separately opt-in with
`b3World_SetMetalFinalization(world, true)` and is off by default.

## Correctness

On the M4 Pro host described by the other 2026-09-01/02 benchmark records, the
randomized 4,096-body direct differential test covered all 22 result floats and
observed `2.29e-05` maximum absolute error. The complete Box3D test executable
passed, including end-to-end solver, recording, determinism, CCD, and
large-world coverage.

## Whole-world result

Paired warm runs used 524,288 unconstrained bodies, four substeps, eight Box3D
workers, and six measured steps per run. Times include the whole
`b3World_Step`, submission, synchronization, and CPU consumption of the shared
finalization result stream.

| Mode | Metal step samples (ms) | Median (ms) |
| --- | --- | ---: |
| Existing fused integration, finalization off | 8.029, 8.299, 8.275 | 8.275 |
| Fused integration plus GPU finalization | 10.688, 10.267, 10.500 | 10.500 |

The experimental stage was about 27% slower at the paired median and was
visibly frequency-sensitive. This is negative evidence, not a release
crossover. The kernel stays available for the next residency/bounds work, but
it is not enabled by `b3World_EnableMetal` alone.

## Reproduction

```sh
cmake --build build/metal-release --target test metal_world_benchmark
./build/metal-release/bin/test
./build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_FINALIZATION=1 ./build/metal-release/bin/metal_world_benchmark
```
