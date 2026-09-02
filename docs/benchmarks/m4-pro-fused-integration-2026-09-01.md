# M4 Pro fused integration - 2026-09-01

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0, Release, `-ffp-contract=off`
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Work: force/damping/gyroscopic velocity integration plus position/quaternion integration, four substeps
- GPU total includes body-property packing, shared-buffer copies, command submission, one wait, and one readback
- CPU primitive reference is serial; the separate whole-world benchmark uses eight CPU workers on both paths

## Fused primitive

All four substeps are encoded into one command buffer. Buffer barriers preserve
substep ordering while state remains on the GPU.

| Bodies | Repeats | Serial CPU ms | GPU total ms | GPU kernel ms | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1500 | 0.003649 | 0.132021 | 0.035657 | 0.028x |
| 128 | 1500 | 0.014335 | 0.134612 | 0.041471 | 0.106x |
| 512 | 1500 | 0.066718 | 0.146075 | 0.048800 | 0.457x |
| 2,048 | 1500 | 0.234525 | 0.168714 | 0.052851 | 1.390x |
| 8,192 | 488 | 0.928883 | 0.242759 | 0.073956 | 3.826x |
| 32,768 | 122 | 3.721596 | 0.551897 | 0.224170 | 6.743x |
| 131,072 | 30 | 15.277444 | 1.772156 | 0.869065 | 8.621x |
| 524,288 | 7 | 61.472290 | 5.808143 | 2.530554 | 10.584x |

This primitive crosses over around 2,048 bodies on the tested machine. The
single-substep implementation crossed around 8,192 bodies, demonstrating that
command fusion and retained GPU state matter more than kernel arithmetic alone.

## Whole-world comparison

This includes the full `b3World_Step` path: solver setup, the fused Metal stage,
CPU finalization, event bookkeeping, and broad-phase maintenance. Both CPU and
Metal worlds use eight Box3D workers. Bodies have no shapes or constraints.

| Bodies | Repeats | CPU ms | Metal ms | Last Metal kernel ms | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 80 | 0.050802 | 0.258120 | 0.048417 | 0.197x |
| 2,048 | 80 | 0.086835 | 0.265257 | 0.050167 | 0.327x |
| 8,192 | 80 | 0.174437 | 0.320192 | 0.067750 | 0.545x |
| 32,768 | 40 | 0.554141 | 0.696210 | 0.326667 | 0.796x |
| 131,072 | 16 | 2.138154 | 2.348977 | 0.730000 | 0.910x |
| 524,288 | 6 | 9.868118 | 8.609667 | 3.101750 | 1.146x |

Whole-world results show that CPU body finalization and current packing/readback
costs consume most primitive gains. GPU integration is therefore still opt-in;
the next end-to-end performance work must move finalization/bounds and more of
the constraint pipeline into the same command stream. GPU frequency varies at
these short durations, so rerun the executable and use multiple trials before
selecting a deployment threshold.

## Reproduction

```sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBOX3D_SAMPLES=OFF -DBOX3D_BENCHMARKS=ON -DBOX3D_UNIT_TESTS=ON
cmake --build build/metal-release --target metal_fused_benchmark metal_world_benchmark
./build/metal-release/bin/metal_fused_benchmark
./build/metal-release/bin/metal_world_benchmark
```
