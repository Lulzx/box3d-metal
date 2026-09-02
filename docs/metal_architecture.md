# Box3D Metal architecture

## Compatibility contract

The CPU implementation at upstream commit `47d7f7cc7e091142c08d11dc7d2e493c5d34f536`
is the behavioral reference. Metal acceleration is opt-in per world and uses the
same public bodies, shapes, joints, events, recording, and query API. CPU and GPU
results are required to be tolerance-equivalent; bit-identical cross-platform
determinism is not promised when Metal is enabled because Metal and CPU floating
point evaluation can differ.

Every GPU stage must have:

- an unmodified CPU reference path;
- a randomized direct differential test, including flags and boundary cases;
- an end-to-end world differential test;
- an explicit fallback if initialization or dispatch fails;
- workload crossover measurements that include synchronization and transfers;
- ABI assertions for every shared structure.

## Current implementation

The first general integrated stage is position/orientation integration. One
thread owns one awake `b3BodyState`. A persistent `MTLBuffer` in shared storage
is grown geometrically and reused. Worlds with constraints outside the
GPU-supported subset use one position dispatch per substep and return to the CPU
constraint solver.

Worlds with no active contacts or joints use a fused ballistic path. It uploads
compact body properties and state once, encodes force, damping, gyroscopic
velocity integration and position/quaternion integration for every substep into
one command buffer, inserts buffer barriers between substeps, waits once, and
reads state back once. The solver orchestrator advances the CPU block
synchronization counters as if both CPU stages ran during every substep.

Contact-only worlds can also keep state GPU-resident through velocity
integration, warm starting, graph-colored contact solve, position integration,
relaxation, and restitution in a single command buffer. The kernels cover normal
impulses, central friction and tangent velocity, twist friction, rolling
resistance, and restitution for both SIMD-wide convex contacts and scalar
multi-manifold mesh contacts. Scalar contacts that spill past the graph-color
budget remain in the same GPU command graph: one Metal thread walks the overflow
array in upstream order once per solver phase, because those constraints may
share bodies and cannot run concurrently. Distance joints (including spring,
limit, and motor modes) and parallel joints use type-dense buffers in the same
colored command graph. A compact descriptor stream preserves upstream order for
mixed distance/parallel overflow joints in one serial kernel per phase. Other
joint types and reaction-threshold events remain on the CPU path. CPU preparation writes constraint and manifold data
directly into persistent `MTLStorageModeShared` memory; the kernels consume
those allocations in place and CPU impulse storage reads them afterward, so
there is no constraint upload or readback copy.

An experimental finalization kernel ports the arithmetic portion of Erin
Catto's body-finalization loop: final rotation, body-origin offset,
farthest-point motion and sleep metrics, and world-space inverse inertia. It
consumes resident solver state in the same command buffer for the fused
unconstrained and fully supported constraint paths. The same command buffer now
continues into shape finalization for awake non-CCD bodies: spheres and capsules
use their exact transformed primitives, while hull and aggregate geometry use
their upstream local bounds. The kernel applies speculative and fat-AABB
margins and emits deterministic flat results in body/shape-list order. CPU code
still owns double-precision outward rounding, CCD, events, sleeping/island
mutation, and dynamic-tree mutation. It is separately opt-in
because reading one 64-byte result per awake shape has not demonstrated a
whole-world win; the next residency boundary is GPU pair generation consuming
these bounds before any compact CPU handoff.

Enable it with:

```c
b3WorldId world = b3CreateWorld(&worldDef);
bool enabled = b3World_EnableMetal(world, minimumBodyCount);
b3World_SetMetalFinalization(world, true); // experimental
b3World_SetMetalBroadPhase(world, true);   // experimental
```

Use `b3World_GetMetalProfile` to verify the selected device, dispatch count,
fallback count, and last GPU execution time. `b3World_DisableMetal` releases the
resources. `minimumBodyCount` is deliberately caller-controlled until benchmark
coverage establishes stable defaults across Apple GPU families.

An independently opt-in pair stage copies Erin's existing static, kinematic,
and dynamic tree topology into persistent shared Metal buffers. A count pass
traverses leaves in upstream stack order. A deterministic 256-lane hierarchical
scan computes stable per-move offsets: SIMD subgroups scan each block, a short
serial kernel prefixes block totals, and a parallel add applies block offsets.
A final traversal writes disjoint candidate ranges in the same command buffer.
The first unexpectedly dense call may submit one write-only retry after growing
the persistent candidate buffer; the steady path submits and waits once. CPU
consumption then reuses the complete upstream callback for moved-proxy
de-duplication, compound children, sensors, filters, joint collision overrides,
and custom user filters. Unsupported threadgroup geometry, tree heights at or
above 63, shader stack overflow, changing counts, allocation failure, or more
than 64 raw candidates per moved proxy on average fall back to the full CPU
traversal before any partial result is consumed.

Broad-phase tree mutation, narrow phase, contact and joint preparation,
unsupported joint solution, continuous collision, events, and sleeping/island
mutation still run on the CPU. Unsupported
constrained worlds retain the position-only path, which may lose to the CPU once
transfer and command-buffer latency are included. These are correct production
paths, not yet the final performance architecture.

| Surface | Current Metal behavior |
| --- | --- |
| Unconstrained velocity and position integration | Fused across all substeps |
| Colored convex and mesh contacts, including friction/rolling/restitution | GPU-resident across all substeps |
| Contact and supported-joint graph overflow | GPU-resident, serial in deterministic constraint order |
| Distance joints, including spring, limit, and motor modes | GPU-resident across all substeps |
| Parallel joints | GPU-resident across all substeps |
| Filter, motor, prismatic, revolute, spherical, weld, or wheel joints; joint reaction-threshold events | CPU constraints plus GPU position stage |
| Broad phase | Experimental opt-in Metal tree traversal; CPU tree mutation, candidate filtering, and contact creation |
| Narrow phase and manifolds | CPU |
| Contact preparation and impulse storage | CPU |
| Body and awake-shape finalization | Experimental opt-in Metal kernels; CPU applies flat results and retains CCD/tree topology |
| CCD, sleeping/island mutation, events, recording, queries | CPU |
| Double-precision world positions | CPU shape fallback preserves outward-rounded far-world AABBs; VF64 exact add/narrow is the candidate GPU seam |
| Cross-platform bit determinism | CPU only; Metal is tolerance-equivalent |

## Target pipeline

The target is one ordered command buffer per world step, with state retained in
private or shared GPU buffers according to measured access patterns:

1. Apply forces, damping, gyroscopic integration, and velocity limits.
2. Update body transforms and bounds.
3. Generate broad-phase pairs using GPU radix sort plus compacted cell/BVH data.
4. Run shape-specialized narrow-phase kernels and compact active manifolds.
5. Build or incrementally update islands and conflict-free constraint batches.
6. Prepare, warm-start, solve, relax, and apply restitution by graph color.
7. Finalize bodies, sleeping candidates, movement events, and readback slices.

The CPU remains responsible for public API mutation between steps, complex
topology changes until they are batched, unsupported geometry paths, recording
serialization, and small workloads below measured crossover points. Unified
memory removes a physical PCIe transfer but does not remove cache-coherency,
resource residency, or CPU/GPU synchronization costs; the design must minimize
ownership changes rather than assume they are free.

## Performance gates

Report cold initialization separately from warm steady state. Measurements must
name the Apple chip, OS, Metal toolchain, build type, body/contact/joint counts,
substeps, CPU worker count, and whether timing includes command submission,
synchronization, and readback. A stage is enabled by default only when an
end-to-end world benchmark demonstrates a repeatable improvement at its selected
threshold without weakening its differential or stress gates.

Run the current primitive benchmark with:

```sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBOX3D_SAMPLES=OFF -DBOX3D_BENCHMARKS=ON
cmake --build build/metal-release --target metal_benchmark
./build/metal-release/bin/metal_benchmark
```

Run the end-to-end convex contact benchmark with
`./build/metal-release/bin/metal_contact_benchmark`. Recorded M4 Pro results and
their variance are in
[`benchmarks/m4-pro-convex-contacts-2026-09-01.md`](benchmarks/m4-pro-convex-contacts-2026-09-01.md).
The corresponding scalar multi-manifold mesh results are in
[`benchmarks/m4-pro-mesh-contacts-2026-09-01.md`](benchmarks/m4-pro-mesh-contacts-2026-09-01.md).
Distance-joint results are in
[`benchmarks/m4-pro-distance-joints-2026-09-01.md`](benchmarks/m4-pro-distance-joints-2026-09-01.md).
Parallel-joint results are in
[`benchmarks/m4-pro-parallel-joints-2026-09-02.md`](benchmarks/m4-pro-parallel-joints-2026-09-02.md).
Experimental finalization results, including the current negative whole-world
performance evidence, are in
[`benchmarks/m4-pro-finalization-2026-09-02.md`](benchmarks/m4-pro-finalization-2026-09-02.md).
The follow-on shape-AABB implementation and result-stream measurements are in
[`benchmarks/m4-pro-shape-finalization-2026-09-02.md`](benchmarks/m4-pro-shape-finalization-2026-09-02.md).
Experimental dynamic-tree traversal results are in
[`benchmarks/m4-pro-pair-generation-2026-09-02.md`](benchmarks/m4-pro-pair-generation-2026-09-02.md).
The follow-on on-device stable scan and compaction correctness checkpoint is in
[`benchmarks/m4-pro-pair-prefix-2026-09-02.md`](benchmarks/m4-pro-pair-prefix-2026-09-02.md).
