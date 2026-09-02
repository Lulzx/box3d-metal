# Performance evidence

## Measurement policy

Performance claims name the chip, OS, compiler, baseline, build, worker count,
body/constraint count, substeps, and timing boundary. Cold initialization is
not mixed into warm steady-state step timing. Whole-world comparisons include
CPU preparation/finalization, command submission, synchronization, and
readback unless explicitly labeled a primitive.

## Test platform

- MacBook Pro `Mac16,8`
- Apple M4 Pro: 12 CPU cores and 16 GPU cores
- 24 GB unified memory
- macOS 26.7 (`25G227`)
- Metal 4 compiler `32023.883`
- AppleClang 21.0.0, Release, `-ffp-contract=off`
- Eight Box3D workers in whole-world benchmarks
- Four substeps

## Summary

| Workload | Crossover/evidence | Largest measured speedup |
| --- | --- | ---: |
| Isolated position integration | Around 131,072 bodies | 1.278x at 524,288 |
| Fused four-substep integration primitive | Around 2,048 bodies | 10.584x at 524,288 |
| Unconstrained whole world | Only largest tested point | 1.146x at 524,288 |
| Convex-contact whole world | Around 262,144 bodies | 1.098x at 262,144 |
| Mesh-contact whole world | Around 8,192 bodies | 1.646x at 131,072 |
| Distance-joint whole world | Near 131,072 bodies | 1.158x at 524,288 |
| Parallel-joint whole world | No stable crossover | 0.974x at 1,048,576 median point |

## Interpretation

Command fusion is the largest demonstrated architectural win: the fused
primitive reaches 10.584x, while full-world bookkeeping reduces that to 1.146x.
Mesh constraints cross earlier than convex-wide stacks in the tested setup.
Distance joints show a large-workload benefit despite compact pack/unpack costs.
Parallel joints expand compatibility but do not yet justify default routing.

The data supports an explicit caller-selected threshold, not a universal
default. GPU frequency, CPU worker scheduling, constraint topology, contact
density, and unsupported stages can move the crossover substantially.

## Reproduce

```sh
./scripts/bootstrap.sh ../box3d-metal-worktree
./scripts/run-benchmarks.sh ../box3d-metal-worktree
```

The benchmark script prints raw CSV-like rows. Run complete executables in at
least three separate processes and compare medians before selecting a threshold.
Detailed original tables are linked from [the documentation index](index.md).
