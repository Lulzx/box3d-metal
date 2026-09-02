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
still owns CCD, events, sleeping/island mutation, and public shape bookkeeping.
Double-precision builds use the exact integer IEEE-754 implementation from
VF64-metal commit `729021777455da72db8809d9ef1269c677d88b3f`: body-center bits
remain binary64 through three round-to-nearest additions, then lower and upper
bounds narrow toward negative and positive infinity respectively. A
scale-aware local-float arithmetic envelope makes the device result
conservative against the CPU `b3ComputeFatShapeAABB` oracle. It is separately opt-in
because reading one 64-byte result per awake shape has not demonstrated a
whole-world win; the remaining boundary is removal of that CPU bookkeeping
stream once all downstream consumers can use resident bounds.

After the first successful resident refit, the next shape dispatch reads its
previous fat bounds directly from the Metal result buffer rather than trusting
the CPU mirror packed into `ShapeInput`. The state is accepted only when both
the shape count and broad-phase revision still match. Buffer growth, proxy
creation/destruction, explicit movement, and tree rebuilds fail closed by
forcing a CPU-oracle reseed. `shapeBoundsResidentDispatchCount` exposes this
transition. This makes the VF64-produced bounds authoritative between eligible
steps, but the buffer is still shared and the full result is still applied to
CPU shapes for current public-query, mesh-contact, sensor, CCD, and fallback
consumers.

The 64-byte result allocation itself is `MTLStorageModePrivate`. On the bounded
resident route—continuous collision disabled, no sensors or existing contacts,
and every awake shape masking out contacts—the command buffer does not encode a
full result blit and the CPU does not execute the flat apply task. The shared
handoff is only the stable 32-byte enlarged subset. `b3Shape_GetAABB` and
`b3Body_ComputeAABB` stage individual 64-byte records on demand. Mutation,
fallback-route expansion, finalization/broad-phase disable, and context disable
perform a checked bulk synchronization; a failed blit keeps the Metal context
and skips the unsafe transition. CPU-oracle recomputation is the single-query
fallback.

`shapeResultApplyCount` distinguishes compatibility-route full applies from
the no-apply resident route, and `shapeBoundsSyncCount` reports how many shape
records were explicitly materialized. This removes the shared full-result
stream and traversal for the bounded route.

The 72-byte shape-input records are also persistent across revision-stable
steps. The CPU checks the exact awake-body id sequence while it already walks
body simulations; a match preserves every cached body index and dispatches the
existing geometry, filter, proxy, local-bound, and resident-fat-bound records
without counting shapes or walking body shape lists. Sleep/wake swap-removal,
tree revision changes, explicit transforms, and filter-only edits invalidate
the registry. A rebuild first materializes any stale CPU mirrors, then repacks
from the CPU oracle. `shapeInputPackCount` and `shapeInputReuseCount` expose the
two routes. Cold/topology rebuilds are still CPU work; maintaining the registry
directly in every topology mutator is a later refinement.

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
and dynamic tree topology into persistent shared Metal buffers. A monotonic
broad-phase revision lets node snapshots remain resident across pair queries.
For supported non-CCD steps, shape finalization writes enlarged leaf bounds
directly into that snapshot and deterministic height-ordered kernels refit
kinematic and dynamic internal nodes. This preserves Erin's topology and DFS
candidate order without a per-step tree upload. CPU create, destroy, explicit
move, unsupported enlargement, and rebuild operations still invalidate the
snapshot before reuse. A count pass
traverses leaves in upstream stack order. A deterministic 256-lane hierarchical
scan computes stable per-move offsets: SIMD subgroups scan each block, a short
serial kernel prefixes block totals, and a parallel add applies block offsets.
A final traversal writes disjoint candidate ranges in the same command buffer.
Each per-move record also carries the resident leaf's shape id and fat AABB, so
the CPU filtering callback does not dereference the CPU dynamic tree for query
metadata. A revisioned resident shape table rejects self/moved-proxy duplicates,
same-body pairs, sensors, and built-in group/category/mask filters during both
the count and write traversals. Shape creation, destruction, and filter mutation
refresh that table; unchanged steps do not repack it. A revisioned mirror of
Erin's 16-byte open-addressed pair-set entries runs the same key construction,
hash finalizer, and linear probing to suppress existing non-compound contacts.
Compound parents stay on the CPU because their contact keys include child ids.
When device leaf update and refit succeed, CPU proxy bookkeeping
consumes a stable GPU-compacted 32-byte record only for each enlarged shape.
A 256-lane hierarchical scan preserves packed shape order across blocks. This
skips the full-result rescan, enlarged-body bit-set reduction, and second
body/shape-list walk. The CPU tree is still enlarged for public queries and
fallback safety.
The first unexpectedly dense call may submit one write-only retry after growing
the persistent candidate buffer; the steady path submits and waits once. CPU
consumption sends ordinary candidates directly to joint/custom filtering and
move-pair append. Compound children retain the complete upstream callback.
Deterministic contact creation remains CPU-owned. Unsupported
threadgroup geometry, tree heights at or
above 63, shader stack overflow, changing counts, allocation failure, or more
than 64 raw candidates per moved proxy on average fall back to the full CPU
traversal before any partial result is consumed.

The narrow-phase route owns a separate persistent geometry registry indexed by
Box3D shape id. On its first dispatch after shape revision, the CPU packs sphere
and capsule endpoints/radii, validates compact supported hulls, deduplicates
identical `b3HullData` content, and writes one point, plane, and triangle stream
plus a 64-byte descriptor for each shape slot. Stable dispatches skip
world-shape traversal and packing. Shape creation, destruction, or geometry mutation invalidates both
this registry and the existing pair metadata revision; filter mutation currently
over-invalidates the geometry registry as a conservative consequence of sharing
that revision. Allocation, count, or hull validation failure rejects the Metal
narrow-phase dispatch before any result is consumed. A separate body-id-indexed
transform registry packs every live static/awake/sleeping body once for the
collision phase. It is keyed by world step, explicit transform revision, and
body-slot count. Same-step dispatches reuse it; body create/destroy/teleport or
the next solved step rebuilds it. Double builds retain all three binary64 world
position bit patterns and perform VF64 subtraction in the shader. Unsupported
contact batches return before either registry is built.

The 16-byte contact input contains eligibility and two shape ids rather than
per-contact geometry or transforms; the MSL kernel follows shape-to-body ids to
load both registries directly.

Broad-phase topology mutation, most narrow-phase shape pairs, contact and joint preparation,
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
| Broad phase | Experimental Metal leaf update, internal refit, stable traversal, and compaction; resident pair records carry query metadata, while CPU topology mutation, filtering, and contact creation remain |
| Narrow phase and manifolds | Sphere-sphere, capsule-sphere, capsule-capsule, and bounded compact hull-sphere local geometry is batched on Metal; compact hull geometry is deduplicated and retained across revision-stable dispatches. CPU applies persistence, materials, callbacks, and state transitions. High-aspect/speculative hull-sphere, other hull pairs, meshes, height fields, and compounds remain CPU |
| Contact preparation and impulse storage | CPU |
| Body and awake-shape finalization | Experimental Metal kernels; private resident bounds feed tree refit and enlarged shapes are stably compacted. Public queries selectively stage requested records; route changes synchronize all bounds. CPU retains CCD/topology |
| CCD, sleeping/island mutation, events, recording, queries | CPU |
| Double-precision world positions | VF64 exact add plus directed narrowing produces conservative far-world AABBs on Metal |
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
The follow-on on-device stable scan, compaction, and unchanged-tree residency
correctness checkpoint is in
[`benchmarks/m4-pro-pair-prefix-2026-09-02.md`](benchmarks/m4-pro-pair-prefix-2026-09-02.md).
Resident leaf refit and VF64 far-world validation are recorded in
[`benchmarks/m4-pro-resident-refit-vf64-2026-09-02.md`](benchmarks/m4-pro-resident-refit-vf64-2026-09-02.md).
Resident moved-proxy and built-in shape filtering are recorded in
[`benchmarks/m4-pro-resident-pair-filtering-2026-09-02.md`](benchmarks/m4-pro-resident-pair-filtering-2026-09-02.md).
Resident existing-contact suppression is recorded in
[`benchmarks/m4-pro-resident-existing-pairs-2026-09-02.md`](benchmarks/m4-pro-resident-existing-pairs-2026-09-02.md).
The first batched sphere-sphere narrow-phase checkpoint is recorded in
[`benchmarks/m4-pro-sphere-narrow-phase-2026-09-02.md`](benchmarks/m4-pro-sphere-narrow-phase-2026-09-02.md).
The capsule extension, including two-point parallel manifolds, is recorded in
[`benchmarks/m4-pro-capsule-narrow-phase-2026-09-02.md`](benchmarks/m4-pro-capsule-narrow-phase-2026-09-02.md).
The bounded hull-sphere extension and its explicit GJK fallback boundary are
recorded in
[`benchmarks/m4-pro-hull-sphere-narrow-phase-2026-09-02.md`](benchmarks/m4-pro-hull-sphere-narrow-phase-2026-09-02.md).
The persistent deduplicated hull-geometry registry and invalidation evidence are
recorded in
[`benchmarks/m4-pro-resident-hull-geometry-2026-09-02.md`](benchmarks/m4-pro-resident-hull-geometry-2026-09-02.md).
The follow-on sphere/capsule residency and 120-byte input checkpoint is recorded
in
[`benchmarks/m4-pro-resident-shape-geometry-2026-09-02.md`](benchmarks/m4-pro-resident-shape-geometry-2026-09-02.md).
The body-transform registry and 16-byte pair-record checkpoint is recorded in
[`benchmarks/m4-pro-resident-body-transforms-2026-09-02.md`](benchmarks/m4-pro-resident-body-transforms-2026-09-02.md).
Private shape results and selective synchronization are recorded in
[`benchmarks/m4-pro-private-shape-results-2026-09-02.md`](benchmarks/m4-pro-private-shape-results-2026-09-02.md).
Persistent shape-input reuse is recorded in
[`benchmarks/m4-pro-shape-input-registry-2026-09-02.md`](benchmarks/m4-pro-shape-input-registry-2026-09-02.md).
