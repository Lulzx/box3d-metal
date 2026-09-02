# M4 Pro distance-joint solver — 2026-09-01

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0, Release, `-ffp-contract=off`
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Workload: independent dynamic-body pairs with spring, lower/upper limits, and motor enabled; four substeps and eight Box3D workers
- Timing: complete `b3World_Step`, including broad-phase maintenance, CPU joint preparation, compact joint packing/unpacking, command submission, synchronization, and body-state readback

## Whole-world comparison

Each row uses median CPU and Metal wall times from three process-level trials.

| Bodies | Joints | Timed steps/trial | CPU median ms | Metal median ms | Speedup from medians |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 40 | 0.124077 | 0.561107 | 0.221x |
| 2,048 | 1,024 | 40 | 0.293545 | 0.663054 | 0.443x |
| 8,192 | 4,096 | 20 | 0.820708 | 0.918327 | 0.894x |
| 32,768 | 16,384 | 8 | 3.100021 | 3.168740 | 0.978x |
| 131,072 | 65,536 | 4 | 14.308761 | 14.354427 | 0.997x |
| 524,288 | 262,144 | 2 | 76.788231 | 66.330315 | 1.158x |

The complete GPU path reaches parity near 131,072 bodies and is `1.158x` faster
at 524,288 bodies on this workload. Small joint sets remain CPU-favorable.
Unlike transient contact constraints, joint simulations live in persistent CPU
graph arrays, so this first joint path packs a compact 204-byte GPU record and
copies four accumulated impulses back once per step. Removing that ownership
transition is a remaining optimization rather than being excluded from timing.

## Correctness gate

The differential suite covers rigid distance constraints, bounded springs,
lower and upper limits, motors, mixed contacts, and a forced 12-joint overflow
set. Across six consecutive four-substep steps, maximum transform error was
`1.07e-6` and maximum linear or angular velocity error was `8.27e-5`. An
unsupported revolute joint separately proves the CPU constraint plus Metal
position fallback.

## Reproduction

```sh
cmake --build build/metal-release --target metal_joint_benchmark
./build/metal-release/bin/metal_joint_benchmark
```
