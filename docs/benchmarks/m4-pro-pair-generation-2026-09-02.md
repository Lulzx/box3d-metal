# M4 Pro GPU pair generation — 2026-09-02

## Scope

This is the first broad-phase slice, not yet a device-resident replacement for
the CPU tree. Metal receives the three existing Box3D dynamic-tree node arrays
and the deterministic move array. A count traversal, stable CPU prefix, and
write traversal reproduce upstream leaf visitation order. The CPU then applies
the unchanged Box3D pair callback and creates contacts in the same order.

The direct oracle used 96 mixed static, kinematic, and dynamic proxies. It
compared all 1,012 raw candidates, including proxy id, tree type, shape id, and
per-move order, exactly. An 80-body fully overlapping case exceeded the
64-candidate-average guard and verified one CPU fallback with zero Metal pair
dispatches. A ten-step contact world exercised GPU traversal plus CPU filtering,
contact creation, Metal shape finalization, and the GPU contact solver.

## Whole-world result

Each body owns one collision-filtered offset sphere. Both modes enable Metal
integration plus body/shape finalization; the experimental mode additionally
enables Metal broad-phase traversal. Three alternating warm processes were
recorded for each mode. Times are complete `b3World_Step` medians.

| Bodies/shapes | Metal pairs off (ms) | Metal pairs on (ms) | Change |
| ---: | ---: | ---: | ---: |
| 512 | 0.287 | 0.594 | +107.1% |
| 2,048 | 0.390 | 0.726 | +86.1% |
| 8,192 | 0.827 | 1.098 | +32.7% |
| 32,768 | 2.842 | 2.751 | -3.2% |
| 131,072 | 11.809 | 11.507 | -2.6% |
| 524,288 | 48.441 | 45.356 | -6.4% |

At 524,288 shapes the last two Metal traversal kernels used about 3.0 ms, while
removing CPU tree traversal saved about 6.1% at the whole-step median. Small
worlds remain dominated by two submissions and an intermediate CPU prefix.

This establishes a large-world traversal crossover, but it does not close the
residency goal. Shape results are still read by the CPU, applied to the CPU
tree, copied back to Metal on the following step, and raw candidates are read
back for filtering. The next implementation must update or rebuild a GPU-owned
broad-phase structure directly from resident shape bounds, perform prefix and
compaction on-device, and return only the CPU-required compact candidate slice.

## Reproduction

```sh
cmake --build build/metal-release --target test metal_world_benchmark
./build/metal-release/bin/test MetalTest
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 \
  ./build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1 \
  ./build/metal-release/bin/metal_world_benchmark
```
