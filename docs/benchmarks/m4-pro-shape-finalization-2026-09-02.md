# M4 Pro GPU shape finalization — 2026-09-02

## Scope

Each awake body owns one offset sphere with collision filtering disabled. Both
paths execute four unconstrained substeps with eight Box3D workers. The baseline
uses Metal fused integration and CPU body/shape finalization. The experimental
path appends body and shape finalization kernels to the command buffer, then the
CPU consumes one flat 64-byte AABB result per shape and updates the dynamic
tree. Timings include the complete `b3World_Step`, synchronization, and CPU
consumption.

The end-to-end oracle covers 2,048 moving spheres, capsules, and offset hulls.
Its worst CPU-versus-GPU AABB component error was `3.81e-06` on an Apple M4 Pro.
The complete float and double-precision Box3D test executables passed. Double
worlds deliberately retain CPU AABB generation because far-world bounds require
outward-rounded binary64 addition before narrowing.

## Whole-world result

Three warm process runs were recorded for each mode. The table compares median
Metal-world step time.

| Bodies/shapes | Finalization off (ms) | Body + shape finalization (ms) | Change |
| ---: | ---: | ---: | ---: |
| 512 | 0.358 | 0.345 | -3.6% |
| 2,048 | 0.409 | 0.443 | +8.4% |
| 8,192 | 0.854 | 1.016 | +19.0% |
| 32,768 | 2.844 | 3.349 | +17.8% |
| 131,072 | 10.402 | 12.754 | +22.6% |
| 524,288 | 44.235 | 52.174 | +17.9% |

This is negative end-to-end evidence. Shape arithmetic has left the CPU, but
ownership has not: the CPU still reads every shape result, walks enlarged
bodies, mutates the dynamic tree, and traverses that tree to generate pairs on
the following step. GPU pair generation must consume resident bounds and return
only compact candidate/contact data before this result stream can be removed.

## Reproduction

```sh
cmake --build build/metal-release --target test metal_world_benchmark
./build/metal-release/bin/test
BOX3D_METAL_SHAPES=1 ./build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 \
  ./build/metal-release/bin/metal_world_benchmark
```
