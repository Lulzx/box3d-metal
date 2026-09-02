# M4 Pro on-device pair compaction — 2026-09-02

## Scope

This checkpoint removes the intermediate CPU prefix from the experimental
Metal broad-phase traversal. The GPU now performs the count traversal, a stable
hierarchical exclusive scan, block-prefix addition, and compact candidate write
inside one command buffer on the steady path. Candidate order remains identical
to Erin Catto's move-array and dynamic-tree traversal order.

The scan uses fixed 256-lane blocks. Metal SIMD subgroups scan within each block,
one short serial kernel prefixes the block totals, and a parallel pass adds each
block offset. A 16-byte summary is the only intermediate result observed by the
CPU. If the exact count exceeds the geometrically retained candidate capacity,
the first command skips writes safely, the CPU grows the buffer, and a single
write-only retry completes the call. Subsequent calls reuse the capacity.

## Correctness gate

The direct CPU oracle contains 607 interleaved static, kinematic, and dynamic
proxies. Its 8,081 raw candidates span three scan blocks, including a partial
last block. The test compares every per-move count and offset and every proxy id,
tree type, shape id, and candidate position exactly.

The deliberately undersized first call required two command buffers: scan plus
one capacity-growth retry. The second call required one command buffer. The
first call uploaded the three tree node arrays; the unchanged second call
uploaded none. An explicit broad-phase bounds invalidation then forced exactly
one fresh tree upload. This proves reuse cannot silently traverse a stale
snapshot. The existing fully overlapping 80-body guard case still rejects the
Metal route and
runs one complete CPU traversal before exposing any candidates. The full Box3D
test suite, focused AddressSanitizer and UndefinedBehaviorSanitizer builds, and
the double-precision Metal test all passed during development.

## Performance status

A naive serial GPU prefix over every move measured about 49 ms at 524,288 moves
and was rejected. The hierarchical replacement was observed around 2.5–4.0 ms
for the complete pair command at that scale during development. No new
whole-world result is accepted here: concurrent CPU and Metal/MPS workloads made
the host unsuitable for a controlled paired measurement. The earlier published
whole-world crossover remains historical evidence for the CPU-prefix version,
not evidence for this implementation.

The remaining residency costs at this checkpoint were structural. Shape results
were still consumed by the CPU, moving broad-phase trees were copied to Metal,
and raw candidates returned to CPU filtering and contact creation. The
subsequent resident-refit checkpoint moved awake leaf updates and internal
refitting onto the GPU; see
[`m4-pro-resident-refit-vf64-2026-09-02.md`](m4-pro-resident-refit-vf64-2026-09-02.md).

## Reproduction

The benchmark accepts selectors so one scale can be measured without running
the complete matrix:

```sh
cmake --build build/metal-release --target test metal_world_benchmark
./build/metal-release/bin/test MetalTest
BOX3D_METAL_WORLD_COUNT=524288 BOX3D_METAL_WORLD_REPEATS=12 \
  BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 \
  BOX3D_METAL_BROAD_PHASE=1 \
  ./build/metal-release/bin/metal_world_benchmark
```

Only publish the timing after checking that the machine is free of competing
CPU, GPU, and MPS workloads and recording alternating pairs-on/pairs-off runs.
