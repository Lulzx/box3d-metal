# M4 Pro convex contact solver — 2026-09-01

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0, Release, `-ffp-contract=off`
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Workload: eight-body-high box stacks, friction `0.6`, rolling resistance `0.05`, four substeps, eight Box3D workers
- Timing: complete `b3World_Step`, including CPU contact preparation/finalization, shared-buffer copies, command submission, wait, and readback

## Whole-world comparison

Each row is the median CPU and Metal wall time from three process-level trials.
GPU frequency and short-run scheduling caused substantial variance; these
results are directional rather than a stable default-threshold claim.

| Bodies | Timed steps/trial | CPU median ms | Metal median ms | Speedup from medians |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 40 | 0.268374 | 1.569554 | 0.171x |
| 2,048 | 40 | 0.549439 | 5.020295 | 0.109x |
| 8,192 | 20 | 0.852600 | 3.793565 | 0.225x |
| 32,768 | 8 | 3.230562 | 9.638860 | 0.335x |
| 131,072 | 4 | 16.699417 | 19.063906 | 0.876x |
| 262,144 | 2 | 37.953770 | 34.570957 | 1.098x |

CPU contact preparation now writes directly into persistent Metal shared memory,
so no wide-constraint upload or readback is needed. The complete contact path
crossed over only at the largest tested workload, reaching `1.098x` at 262,144
bodies. CPU finalization, body-state copies, and per-color barriers still
dominate. The path therefore remains opt-in and governed by the caller's body
threshold. The next performance work should retain body data across world steps,
reduce graph-color barriers, and move finalization/bounds onto the GPU before
considering a default.

## Correctness gates

- A 128-body, eight-layer stack exercises static and dynamic contacts, friction,
  tangent velocity, twist friction, and rolling resistance across multiple graph
  colors for ten consecutive steps. Maximum transform error was `4.77e-7`; the
  maximum linear/angular velocity error was `3.98e-6`.
- A 64-sphere bounce test exercises restitution and rolling resistance. It
  produced upward velocity above `2.18 m/s`, with maximum transform error
  `1.19e-7` and velocity error `2.38e-7`.

## Reproduction

```sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBOX3D_METAL=ON -DBOX3D_SAMPLES=OFF -DBOX3D_BENCHMARKS=ON
cmake --build build/metal-release --target metal_contact_benchmark
./build/metal-release/bin/metal_contact_benchmark
```
