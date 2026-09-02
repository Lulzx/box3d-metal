# M4 Pro mesh contact solver - 2026-09-01

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0, Release, `-ffp-contract=off`
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Workload: independent boxes on a two-triangle static mesh, friction `0.6`, rolling resistance `0.05`, four substeps, eight Box3D workers
- Timing: complete `b3World_Step`, including collision maintenance, CPU constraint preparation/finalization, command submission, synchronization, and body-state readback

## Whole-world comparison

Each row uses median CPU and Metal wall times from three process-level trials.

| Bodies | Timed steps/trial | CPU median ms | Metal median ms | Speedup from medians |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 40 | 0.183037 | 0.428761 | 0.427x |
| 2,048 | 40 | 0.458993 | 0.744328 | 0.617x |
| 8,192 | 20 | 1.523123 | 1.247577 | 1.221x |
| 32,768 | 8 | 5.444484 | 4.695943 | 1.159x |
| 131,072 | 4 | 21.287510 | 12.937166 | 1.646x |

The mesh path crosses over near 8,192 contacts on this workload and reaches
`1.646x` at 131,072 bodies. Scalar multi-manifold constraints are prepared
directly in persistent shared Metal buffers and solved alongside SIMD-wide
convex contacts by graph color. Small workloads remain CPU-favorable because
command submission and synchronization dominate.

## Correctness gate

The differential test combines a mesh-contact bottom layer and convex-contact
upper layer for ten consecutive steps, with friction, rolling resistance, and
restitution enabled. Maximum transform error was `4.77e-7`; maximum linear or
angular velocity error was `3.6e-6`.

## Reproduction

```sh
cmake --build build/metal-release --target metal_mesh_benchmark
./build/metal-release/bin/metal_mesh_benchmark
```
