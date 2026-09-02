# M4 Pro parallel-joint solver — 2026-09-02

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0, Release, `-ffp-contract=off`
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Workload: independent dynamic-body pairs with soft parallel-axis alignment and torque limiting; four substeps and eight Box3D workers
- Timing: complete `b3World_Step`, including broad-phase maintenance, CPU joint preparation, compact joint packing/unpacking, command submission, synchronization, and body-state readback

## Whole-world comparison

Each row uses median CPU and Metal wall times from three process-level trials.

| Bodies | Joints | Timed steps/trial | CPU median ms | Metal median ms | Speedup from medians |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 40 | 0.109306 | 0.326899 | 0.334x |
| 2,048 | 1,024 | 40 | 0.185069 | 0.304244 | 0.608x |
| 8,192 | 4,096 | 20 | 0.435473 | 0.514429 | 0.846x |
| 32,768 | 16,384 | 8 | 1.218839 | 1.206859 | 1.010x |
| 131,072 | 65,536 | 4 | 5.058291 | 6.436219 | 0.786x |
| 524,288 | 262,144 | 2 | 24.120687 | 25.841167 | 0.933x |
| 1,048,576 | 524,288 | 1 | 49.393044 | 50.713375 | 0.974x |

The isolated near-parity result at 32,768 bodies did not persist at larger
sizes, so this workload has no demonstrated stable whole-world crossover. The
GPU solve itself scales, but CPU preparation, compact-record ownership changes,
body finalization, and synchronization consume the gain. Parallel-joint support
therefore expands the compatible GPU-resident constraint surface without
justifying a default threshold. Callers should retain an explicit threshold and
measure their own workload.

## Correctness gates

- A disjoint mixed-type graph alternates distance and parallel joints for ten
  consecutive four-substep steps. Maximum transform error was `7.45e-9` and
  maximum linear or angular velocity error was `1.79e-7`.
- A shared-hub graph forces 12 mixed distance/parallel joints into overflow.
  The ordered serial descriptor kernel produced maximum transform error
  `1.82e-6` and velocity error `3.29e-4`, with zero GPU fallbacks.
- A mixed distance/parallel case anchored to static bodies verifies the shader's
  dummy-state path; its maximum transform error was `4.47e-8`.

## Reproduction

```sh
cmake --build build/metal-release --target metal_parallel_joint_benchmark
./build/metal-release/bin/metal_parallel_joint_benchmark
```
