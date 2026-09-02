# M4 Pro position integration baseline - 2026-09-01

This is an early primitive benchmark, not a whole-engine performance claim.

## Provenance

- Host: MacBook Pro `Mac16,8`
- SoC: Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 16 GPU cores
- Unified memory: 24 GB
- OS: macOS 26.7 (`25G227`)
- Metal: Metal 4, compiler `32023.883`
- Compiler: AppleClang 21.0.0
- Upstream baseline: `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
- Build: CMake Release, Ninja, `-ffp-contract=off`
- State size: 56 bytes per awake body
- GPU timing: warm, synchronous, includes CPU-to-shared-buffer copy, command submission, wait, and shared-buffer-to-CPU copy
- CPU timing: serial reference integration loop

## Results

| Bodies | Repeats | CPU ms | GPU total ms | GPU kernel ms | End-to-end speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2000 | 0.000112 | 0.100351 | 0.011081 | 0.001x |
| 128 | 2000 | 0.000418 | 0.102160 | 0.011763 | 0.004x |
| 512 | 2000 | 0.001626 | 0.098572 | 0.008330 | 0.016x |
| 2,048 | 2000 | 0.006688 | 0.094590 | 0.003847 | 0.071x |
| 8,192 | 976 | 0.027992 | 0.120425 | 0.015520 | 0.232x |
| 32,768 | 244 | 0.111827 | 0.191694 | 0.036629 | 0.583x |
| 131,072 | 61 | 0.443168 | 0.406786 | 0.109512 | 1.089x |
| 524,288 | 15 | 1.784119 | 1.396439 | 0.391122 | 1.278x |

The measured crossover for this isolated stage is around 131,072 bodies. The
roughly 0.09-0.10 ms fixed GPU cost at small workloads makes a per-stage,
synchronous design inappropriate for normal game-sized worlds. The next
performance milestone is therefore stage fusion and GPU residency across
velocity integration, contacts, constraint iterations, and position integration.

## Reproduction

```sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBOX3D_SAMPLES=OFF -DBOX3D_BENCHMARKS=ON -DBOX3D_UNIT_TESTS=ON
cmake --build build/metal-release --target metal_benchmark
./build/metal-release/bin/metal_benchmark
```
