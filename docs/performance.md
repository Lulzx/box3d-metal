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
| Experimental GPU finalization | No crossover | 27% slower at 524,288 bodies |
| GPU shape finalization | No crossover | 17.9% slower at 524,288 shapes |
| Experimental GPU tree traversal | Around 32,768 shapes | 1.068x at 524,288 shapes |

## Interpretation

Command fusion is the largest demonstrated architectural win: the fused
primitive reaches 10.584x, while full-world bookkeeping reduces that to 1.146x.
Mesh constraints cross earlier than convex-wide stacks in the tested setup.
Distance joints show a large-workload benefit despite compact pack/unpack costs.
Parallel joints expand compatibility but do not yet justify default routing.
The finalization arithmetic kernel is correct and fused into the resident
command graph, but its additional shared result stream made the 524,288-body
paired median 10.500 ms versus 8.275 ms with finalization disabled. It is
therefore an experimental opt-in, not a default optimization.

The follow-on shape kernel is also correct but does not yet change ownership:
at 524,288 bodies with one sphere each, the paired three-process median was
52.174 ms with body/shape finalization versus 44.235 ms with finalization off,
a 17.9% regression. The CPU still reads every 64-byte shape result, mutates the
dynamic tree, and traverses it for pairs. GPU pair generation must consume
resident bounds before this stage can remove that stream.

The first Metal dynamic-tree traversal slice produced an exact raw candidate
stream but copied the CPU tree, used an intermediate CPU prefix, and returned
candidates to CPU filtering. Against the same Metal finalization configuration,
that historical version improved the paired 524,288-shape median from 48.441 ms
to 45.356 ms (6.4%) while adding 86-107% below 2,048 shapes. Its two kernels used
about 3.0 ms at the largest point.

The current implementation replaces that CPU prefix with a deterministic
hierarchical Metal scan and compact write. A 607-proxy oracle compares all
8,081 candidates and their per-move order exactly; steady state uses one command
buffer. The follow-on resident-tree path updates enlarged leaves and refits
internal nodes on-device; a ten-step contact world uses one initial upload and
ten refits. VF64 also removes the double-precision shape fallback with
far-world CPU-oracle containment. No new whole-world number is accepted because concurrent CPU and MPS
loads contaminated the available host. The earlier crossover remains evidence
for the traversal architecture, not a benchmark of the current scan.

The data supports an explicit caller-selected threshold, not a universal
default. GPU frequency, CPU worker scheduling, constraint topology, contact
density, and unsupported stages can move the crossover substantially.

## Reproduce

```sh
./scripts/bootstrap.sh ../box3d-metal-worktree
./scripts/run-benchmarks.sh ../box3d-metal-worktree
BOX3D_METAL_SHAPES=1 ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_WORLD_COUNT=524288 BOX3D_METAL_WORLD_REPEATS=12 \
  BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
```

The benchmark script prints raw CSV-like rows. Run complete executables in at
least three separate processes and compare medians before selecting a threshold.
Detailed original tables are linked from [the documentation index](index.md).
