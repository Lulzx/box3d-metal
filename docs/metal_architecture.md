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
those allocations in place, so there is no constraint upload or readback copy.
CPU-prepared and fallback contacts retain upstream impulse storage, while
complete resident convex sets leave post-solve impulses in the contact-ID table.

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
steps. The result buffer is private, and bounded resident routes leave both
shape bounds and the refitted broad-phase snapshot on-device. Public queries
and unsupported mesh-contact, sensor, CCD, mutation, and fallback consumers
synchronize the CPU oracle explicitly.

The 64-byte result allocation itself is `MTLStorageModePrivate`. On the bounded
resident route—continuous collision disabled, no sensors, and either every
awake shape masking out contacts or the current collision phase proving complete
zero-exception resident convex coverage with the Metal broad phase—the command
buffer does not encode a full result blit and the CPU does not execute the flat
apply task. The stable enlarged subset remains private and is consumed directly
by the next Metal pair query.
`b3Shape_GetAABB` and
`b3Body_ComputeAABB` stage individual 64-byte records on demand. Mutation,
fallback-route expansion, finalization/broad-phase disable, and context disable
perform a checked bulk synchronization; a failed blit keeps the Metal context
and skips the unsafe transition. CPU-oracle recomputation is the single-query
fallback.

`shapeResultApplyCount` distinguishes compatibility-route full applies from
the no-apply resident route, and `shapeBoundsSyncCount` reports how many shape
records were explicitly materialized. This removes the shared full-result
stream and flat result traversal for the bounded route. Body finalization also
skips each pointer-linked CPU shape list whenever Metal produced a complete
awake-shape result set. Compatibility routes apply the flat result after body
bookkeeping; private routes leave the CPU AABB mirror stale until a query or
fallback explicitly synchronizes it. `finalizationShapeTraversalBypassCount`
reports those phases. On the unconstrained route and the contact-only route with
complete resident sphere/capsule/hull-sphere or admitted canonical box-pair collision and preparation coverage,
the entire contiguous CPU body bookkeeping walk is also omitted. Mesh, overflow,
joint, callback, hit-event, topology-exception, sleep, and CCD routes retain it.

The 72-byte shape-input records are also persistent across revision-stable
steps. Every awake-set membership or ordering mutation advances a monotonic
world revision; the cache stores that revision with its body count. A match
preserves every cached body index in constant time and dispatches the existing
geometry, filter, proxy, local-bound, and resident-fat-bound records without
scanning body ids, counting shapes, or walking body shape lists. Create,
destroy, sleep/wake swap-removal, awake-set transfers, snapshot restore, tree
revision changes, explicit transforms, and filter-only edits invalidate the
appropriate registry authority. A rebuild first materializes stale CPU
mirrors, then repacks from the CPU oracle. `shapeInputPackCount`,
`shapeInputReuseCount`, and `shapeInputOrderRevisionCheckCount` expose the two
routes and their O(1) authorization. Cold/topology rebuilds remain CPU work.

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

Body finalization also avoids awake-island scratch traffic when world sleeping
is disabled. In that mode no later stage consumes the per-worker island bitsets,
so finalization neither clears them nor performs the per-body island lookup and
bit write. `awakeIslandBitSetClearBypassCount` and
`lastAwakeIslandBitSetBytes` expose this boundary. Sleep-enabled worlds retain
Erin's original clear, deterministic union, split-candidate reduction, and
reverse island traversal unchanged.

Finalization arithmetic now lands in a private Metal buffer. Shape AABB
generation and resident tree refit consume that private authority directly;
they no longer share a CPU-visible result allocation. A checked blit populates
a separate host mirror when sleeping or continuous collision requires fields
from the device result. On the stable sleep-disabled, non-CCD route, that blit
is omitted: the remaining CPU body pass recomputes Erin's finalization
arithmetic from the returned solver states while the private device result
continues directly into shape AABB generation and tree refit.
`lastFinalizationReadbackBytes` reports the retained transfer and
`finalizationReadbackBypassCount` reports successful zero-readback steps. This
removes the 100-byte-per-body finalization stream on the bounded route. The
unconstrained resident route and fully resident contact-only route also retain
the finalization-property centers, cleared forces/torques, inverse inertia,
transient flags, and absolute body transforms on-device, so they omit the
remaining CPU body traversal as well.
`finalizationBodyTraversalBypassCount` reports successful omissions.

The same kernel now writes a 72-byte private move-event record in deterministic
awake-sim order. It snapshots user data and body identity and carries both the
float compatibility position and VF64-authored exact binary64 position bits.
The remaining CPU body pass deliberately leaves the public event array
untouched. `b3World_GetBodyEvents` is the materialization boundary: only that
query blits the private records and decodes them into Box3D's public ABI. A
world step that overwrites an unobserved prior event stream performs no event
readback. `bodyMoveEventDispatchCount`, `bodyMoveEventCpuWriteBypassCount`,
`bodyMoveEventSyncCount`, and `lastBodyMoveEventReadbackBytes` expose the
boundary. Sleep and CCD retain Erin's original CPU event path because their
event count and `fellAsleep` bookkeeping are not dense. A forced sleep between
the step and the public query first materializes the private stream so Erin's
existing transition can mark the matching event without changing semantics.

On that bounded route, finalization also scatters each awake body's completed
rotation and origin position into the body-id-indexed narrow-phase transform
registry. Double-precision builds preserve the position as VF64 binary64 bits;
the narrow-phase kernel continues to subtract those bits before narrowing the
relative displacement. Static and sleeping records are seeded from the CPU
only when body topology or the explicit transform revision changes. A
successful command marks the registry authoritative for the current world
step, so the next collision pass can consume it without repacking CPU body
sims. `narrowPhaseTransformDeviceRefreshCount` exposes the device updates.
Explicit transforms, topology changes, unsupported routes, sleeping, and CCD
fail closed through the existing revision/step checks. On the bounded
unconstrained and fully resident contact-only routes this registry is also the
authority for lazy public body transforms: the first CPU consumer reconstructs
the awake `b3BodySim` mirror once, while unobserved steps perform no CPU body
walk. A stable cached contact-input revision lets collision consume that registry
without first touching the stale CPU sim array; a registry rebuild, Metal
failure, or compacted CPU exception synchronizes body sims and shape bounds
before Erin's collision task runs.

The same bounded finalization kernel resets resident solver position/rotation
deltas and transient state flags after publishing their absolute transform and
flags. The CPU bookkeeping pass reconstructs its motion metrics from the old
CPU pose and the device-authored absolute pose, so event/debug behavior remains
oracle-compatible even though the returned state deltas are already ready for
the next step. Before the following solver command, a monotonic world revision
detects every public velocity, impulse, lock, wake/order, snapshot, explosion,
or CPU-solver mutation. A matching count and revision omit the CPU-to-device
state copy; a mismatch performs the full upload. This removes the former
whole-array `memcmp` from stable steps. `bodyStateUploadCount`,
`bodyStateReuseCount`, `lastBodyStateUploadBytes`, and
`bodyStateRevisionCheckCount` expose the gate. On the bounded unconstrained and
fully resident contact-only, sleep-disabled, non-CCD routes, the private Metal
state array remains authoritative across steps. Public velocity queries, state
mutations, awake-set topology changes, recording, Metal shutdown, and any CPU
collision/solver route materialize the full awake array under a dedicated
synchronization lock. This preserves multithreaded read-only query safety and
fail-closed fallback behavior while removing the unconditional solved-state copy
from the steady step.
`bodyStateSyncCount` and `lastBodyStateReadbackBytes` expose those lazy
boundaries. Mixed constrained Metal solves still copy solved states eagerly.

The awake `b3BodySim` array follows the same ownership rule on the bounded
unconstrained and fully resident contact-only routes. Device finalization advances its resident center in VF64
binary64 when enabled, publishes the absolute transform and sleep metric,
updates inverse inertia, and clears force, torque, and transient state. Public
transform/property consumers and route changes reconstruct the full CPU mirror
under the state synchronization lock. `bodySimSyncCount` and
`lastBodySimSyncCount` expose these boundaries; a zero value together with a
`finalizationBodyTraversalBypassCount` increment proves an unobserved step did
not execute Erin's per-body CPU finalization loop.

The adjacent 128-byte integration-property stream is revision-resident on the
same bounded route. Device finalization writes the absolute quaternion and
inverse world inertia back into it and clears force and torque, so an unchanged
awake ordering needs no CPU body-sim traversal or property upload on the next
step. Body creation/destruction, wake/sleep/enable transfers, explicit
transforms, mass/inertia, damping, gravity, wind, force, and torque mutations
advance a world revision and fail closed to a full repack. The profile exposes
`bodyPropertyUploadCount`, `bodyPropertyReuseCount`, and
`lastBodyPropertyUploadBytes`; unsupported/fallback routes do not preserve
property authority.

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
When device leaf update and refit succeed, a stable private record is compacted
for each enlarged shape. A 256-lane hierarchical scan preserves packed shape
order across blocks. The next pair dispatch consumes those proxy keys and fat
bounds directly, and an epoch-tagged kernel marks moved leaves without clearing
or uploading a CPU move list. This skips the full-result rescan,
enlarged-body bit-set reduction, second body/shape-list walk, CPU proxy
enlargement traversal, and move-list round trip. Public queries, CPU mutation,
route changes, and dispatch failure conservatively restore every moving CPU
leaf and repopulate Erin's move set before returning to the CPU oracle.
The first unexpectedly dense call may submit one write-only retry after growing
the persistent candidate buffer; the steady path submits and waits once. Each
move record also carries a residual-filter bit. A second stable scan compacts
only move indices involving custom filtering or a compound target. Joint create,
destroy, and `collideConnected` mutation revise a deduplicated device hash set
of body pairs for which any joint disables collision. Count and write traversals
reject those exact pairs on-device, so unrelated and collision-enabled joints
do not force a whole candidate range through `b3ShouldBodiesCollide`.
Ordinary candidate ranges bypass the CPU filtering task and fixed move-pair
allocation; the serial commit consumes each range in reverse traversal order,
exactly matching Erin's prepend-list creation order. Exception moves retain
custom callbacks, compound-child traversal, and the existing move-pair path.
Registry allocation or validation failure returns to the CPU oracle before a
partial plan is consumed. Deterministic contact creation and all coupled contact,
body-edge, solver-set, pair-set, event, and island topology remain CPU-owned. Unsupported
threadgroup geometry, tree heights at or
above 63, shader stack overflow, changing counts, allocation failure, or more
than 64 raw candidates per moved proxy on average fall back to the full CPU
traversal before any partial result is consumed.

The narrow-phase route owns a separate persistent geometry registry indexed by
Box3D shape id. On its first dispatch after shape revision, the CPU packs sphere
and capsule endpoints/radii, validates compact supported hulls, deduplicates
identical `b3HullData` content, and writes one point, plane, and triangle stream
plus a 96-byte geometry/material descriptor for each shape slot. Stable dispatches skip
world-shape traversal and packing. Shape creation, destruction, or geometry mutation invalidates both
this registry and the existing pair metadata revision; material mutation has a
separate revision, while filter mutation currently
over-invalidates the geometry registry as a conservative consequence of sharing
that revision. Allocation, count, or hull validation failure rejects the Metal
narrow-phase dispatch before any result is consumed. A separate body-id-indexed
transform registry packs every live static/awake/sleeping body transform and
local center once for the collision phase. It is keyed by world step, explicit transform revision, and
body-slot count. Same-step dispatches reuse it; body create/destroy/teleport or
the next solved step rebuilds it. Double builds retain all three binary64 world
position bit patterns and perform VF64 subtraction in the shader. Unsupported
contact batches return before either registry is built.

The 40-byte contact input contains eligibility, two shape ids, contact identity,
contact generation, and a CPU-seeded hull SAT feature rather than per-contact
geometry or transforms. A cache
key combines pair-set, constraint-graph, and explicit eligibility revisions.
When it is unchanged, the input/order buffer is reused without gathering graph
contact IDs or writing CPU records. The MSL kernel follows shape-to-body ids to
load current transforms, body indices, and transient fast flags directly.

Full 240-byte narrow-phase outputs are private. A deterministic 256-lane block
scan, serial block prefix, and parallel scatter classify stable resident
contacts separately from ordered CPU exceptions in the same command buffer.
Every supported result is finalized into the private contact-ID table, while
only callback, topology, first-touch, unsupported, and other fail-closed
exceptions enter the shared stream. Each exception carries its contact ID and
remains ordered by the original contact-array index. CPU collision workers
consume that compact list directly; unchanged stable resident steps return zero
shared manifold bytes and schedule no collision task.
The scatter pass rotates active normals, constructs both COM-relative anchors
with VF64 translation subtraction, resolves default material parameters and
rotated tangent velocity, and feature-matches the prior impulse table. CPU
application skips matrix construction, local-to-world transforms, origin-to-COM
adjustment, and its old per-point persistence search.

The same scatter writes an identical finalized record to a persistent private
table indexed by Box3D contact id. Compact exceptions remain ordered by
awake-contact input index, while the private copy sets `inputIndex` to the
contact id so its address and identity are independent of input permutation.
Canonical box contacts carry their SAT separation, feature type, and feature
indices in this table. Face-A and face-B cache hits rebuild the clipped manifold
before fresh SAT, and later dispatches consume the private copy even when a
topology revision repacks the shared input registry. Edge-pair caches currently
fall through to fresh GPU SAT; high-aspect boxes remain CPU-owned until their
rotating cached-face acceptance matches the CPU oracle.
The table is exposed only through an explicit diagnostic/fallback blit. Entries
are authoritative only for contacts marked eligible in the current successful
dispatch; stale unsupported slots are never consumed.

For an already-touching, non-fast contact whose current record was refreshed by
the device, no collision worker runs. The contact-ID table remains
authoritative, and a world generation makes its CPU manifold a lazy mirror
without writing a stale flag on every stable contact. First touches,
separations, hit-event contacts, recording, callbacks, recycling, CCD, and
topology transitions continue through Erin's CPU contact path as compact
exceptions. If the collision graph revision remains unchanged and every convex
graph contact was classified stable, the dispatch also proves complete
resident ownership. Solver setup then uses the known count instead of walking
contacts to recheck their ownership flags. A body-index swap does not invalidate
the input registry because the preparation scatter reads the current index from
the per-step body table; a newly fast body still forces a CPU exception through
that same table.

The same zero-exception proof defers each worker's contact-state bitset clear.
No collision worker can write state bits in that phase, so worker union and the
serial state-change traversal are skipped as well. Stale storage is never
observed: any later callback, CCD, first-touch, separation, unsupported contact,
or Metal fallback clears all worker bitsets to the current contact-ID capacity
before dispatching CPU collision work. Diagnostic manifold and SAT counters are
reset and aggregated independently of the bitsets.

Solver hit-event scratch follows the same rule. Before worker launch, the
current narrow-phase compact event-ID count is known independently of the prior
post-solve table. A complete resident convex set with no mesh/overflow contacts
and an empty event list defers all contact-capacity hit-event bitset clears.
Successful Metal solve/store leaves the stale bits unreachable because no
worker can set `hasHitEvents`. An event-enabled contact clears normally; if the
Metal solver rejects after deferral, the orchestrator clears every worker
bitset before advancing any CPU store stage.

Public contact/body/shape queries, force debug drawing, snapshots, sleep
transitions, Metal disable, and CPU solver fallback materialize stale geometry
by contact ID and contact generation. CPU solver fallback first attempts one
bulk table blit; if that fails, it recomputes only stale convex manifolds with
the CPU oracle before preparation. Sleeping an island aborts if its required
mirror cannot be synchronized, and Metal disable retains the context on a
readback failure.

That authority now survives into solver setup explicitly. The scatter kernel
feature-matches the prior generation-tagged impulse table, emits persistence and
normal warm-start values, computes default material mixing and tangent velocity,
and finalizes both center-of-mass-relative anchors. Each collision worker clears
the transient ownership bit before overlap/recycling decisions and sets it only
after the resident result passes through Box3D's allocation, callback, and
topology path. Contacts requesting a pre-solve callback remain CPU
owned. Solver setup counts marked contact ids in graph-color order and exposes
SIMD-wide coverage only when every colored convex contact is resident-table
authoritative and no convex overflow exists.

That complete-coverage gate now drives a Metal contact-preparation kernel at the
front of the existing solver command buffer. CPU workers skip
`b3PrepareContacts_Convex`; instead, the kernel reads normal and identity from
the private contact-id table and writes the existing 1,696-byte SIMD-wide
constraint ABI. Finalized anchors, point persistence, default materials, tangent
velocity, and normal warm starts come from the private result. Body indices,
custom callback results, contact-scope impulses, and manifold storage identity are retained in a
generation-tagged 224-byte shared table indexed by contact ID. The CPU seeds a
record when a contact first becomes eligible. On later generation-stable steps,
the manifold scatter validates the prior impulse and preparation records, then
refreshes the table in place with current body indices, finalized anchors and
materials, contact-scope impulses, feature IDs, persistence, and normal warm
starts. Collision workers retain disjoint writes for recycling, pre-solve,
custom material callbacks, and first-touch records. Solver submission
then initializes tail lanes and bulk-copies one four-byte contact-ID schedule per
active color; it does not dereference contacts or repack their metadata. The
kernel computes Erin's tangent frame,
softness, effective normal and tangent masses, friction centers, lever arms,
relative velocities, twist mass, rolling mass, and projected warm-start
impulses on Metal. A status word validates table authority without staging the
table. If input packing or a later unsupported constraint rejects the route,
CPU convex preparation is rerun before the CPU solver fallback. This removes
CPU preparation arithmetic plus the dedicated solver-time contact traversal and
144-byte lane stream. CPU manifold allocation, custom callbacks, exceptional table writes,
the graph-color schedule, events, and topology remain.
The 112-byte post-solve record carries contact generation plus each point's
feature ID. The next manifold scatter validates contact identity and generation,
claims matching features in upstream order, and writes normal impulses and
persistence bits into the 240-byte finalized record. Friction, twist, and rolling
warm-start terms are restored at contact scope. A recycled contact slot with a
different generation cannot inherit the old result.
Successful resident solves now skip the all-contact CPU impulse-store walk.
During the already-required narrow-phase input pack, Box3D retains only IDs whose
shapes requested hit events. The store stage processes that compact exception
list on one worker, synchronizes qualifying contacts by contact generation and
feature ID, and leaves ordered event construction unchanged. With no hit-event
requests, the stage touches no contacts. Public contact/body/shape queries sync
only requested records; force debug drawing and snapshots explicitly sync the
contacts they consume. CPU solver routes invalidate old result authority before
fallback, so an earlier GPU result cannot overwrite a newer CPU manifold.

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
| Broad phase | Experimental Metal leaf update, internal refit, private resident move-list consumption, stable traversal, candidate planning, exact blocked-joint body-pair rejection, and residual-filter move compaction. Ordinary ranges bypass CPU candidate filtering; custom/compound exceptions and deterministic contact topology creation remain CPU-owned |
| Narrow phase and manifolds | Sphere-sphere, capsule-sphere, capsule-capsule, bounded compact hull-sphere geometry, and canonical box pairs are finalized on Metal. Box hulls may be dynamic-dynamic and have unequal extents, but each hull is bounded to a 16:1 maximum/minimum extent ratio. The box path includes face SAT/clipping/four-point reduction and Gauss-valid edge contacts. Stable touching contacts emit no shared manifold record, run no CPU collision worker, reuse the resident input/order registry, and bypass per-contact solver coverage checks. Ordered callback/topology/first-touch exceptions retain the CPU path. Lazy CPU mirrors synchronize only at explicit boundaries. CPU retains cold/revision packing, manifold allocation, callbacks, recycling, and state transitions. High-aspect/speculative hull-sphere, high-aspect boxes, other hull pairs, meshes, height fields, and compounds remain CPU |
| Contact preparation and impulse storage | Complete colored resident convex sets are prepared on Metal. The contact-ID lane schedule remains resident across unchanged constraint-graph revisions. After restitution, Metal extracts a 112-byte result per active contact. The all-contact CPU store is bypassed; hit-enabled exceptions and explicit public/debug/snapshot consumers synchronize individual records. Mixed/recycled/callback/overflow sets remain CPU |
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
Private full manifold results and active-only deterministic readback are
recorded in
[`benchmarks/m4-pro-private-manifold-results-2026-09-02.md`](benchmarks/m4-pro-private-manifold-results-2026-09-02.md).
Fused manifold orientation finalization is recorded in
[`benchmarks/m4-pro-manifold-finalization-2026-09-02.md`](benchmarks/m4-pro-manifold-finalization-2026-09-02.md).
The contact-id-indexed private manifold table is recorded in
[`benchmarks/m4-pro-resident-manifold-table-2026-09-02.md`](benchmarks/m4-pro-resident-manifold-table-2026-09-02.md).
The solver ownership gate is recorded in
[`benchmarks/m4-pro-resident-solver-ownership-2026-09-02.md`](benchmarks/m4-pro-resident-solver-ownership-2026-09-02.md).
The first resident convex contact-preparation kernel is recorded in
[`benchmarks/m4-pro-resident-contact-preparation-2026-09-02.md`](benchmarks/m4-pro-resident-contact-preparation-2026-09-02.md).
The follow-on contact-ID metadata residency and four-byte lane schedule are
recorded in
[`benchmarks/m4-pro-contact-prepare-residency-2026-09-02.md`](benchmarks/m4-pro-contact-prepare-residency-2026-09-02.md).
Compact post-solve impulse extraction and hit-event equivalence are recorded in
[`benchmarks/m4-pro-contact-impulse-residency-2026-09-02.md`](benchmarks/m4-pro-contact-impulse-residency-2026-09-02.md).
Constraint-graph revisioning and unchanged-step schedule reuse are recorded in
[`benchmarks/m4-pro-resident-contact-schedule-2026-09-02.md`](benchmarks/m4-pro-resident-contact-schedule-2026-09-02.md).
Feature-matched resident warm-start carry is recorded in
[`benchmarks/m4-pro-resident-warm-start-2026-09-02.md`](benchmarks/m4-pro-resident-warm-start-2026-09-02.md).
Lazy public-manifold synchronization and the compact hit-event exception path
are recorded in
[`benchmarks/m4-pro-lazy-contact-impulse-sync-2026-09-02.md`](benchmarks/m4-pro-lazy-contact-impulse-sync-2026-09-02.md).
GPU-authored persistence, COM-relative anchors, and default material
finalization are recorded in
[`benchmarks/m4-pro-contact-finalization-residency-2026-09-02.md`](benchmarks/m4-pro-contact-finalization-residency-2026-09-02.md).
Direct device refresh of stable contact-preparation records is recorded in
[`benchmarks/m4-pro-device-contact-prepare-refresh-2026-09-02.md`](benchmarks/m4-pro-device-contact-prepare-refresh-2026-09-02.md).
Stable collision-application bypass and lazy manifold geometry synchronization
are recorded in
[`benchmarks/m4-pro-resident-collision-bypass-2026-09-02.md`](benchmarks/m4-pro-resident-collision-bypass-2026-09-02.md).
GPU exception compaction, zero-byte stable manifold output, and zero-contact
steady collision tasks are recorded in
[`benchmarks/m4-pro-contact-exception-compaction-2026-09-02.md`](benchmarks/m4-pro-contact-exception-compaction-2026-09-02.md).
Revisioned contact input/order reuse and solver coverage proof are recorded in
[`benchmarks/m4-pro-contact-input-residency-2026-09-02.md`](benchmarks/m4-pro-contact-input-residency-2026-09-02.md).
Zero-exception contact-state traversal bypass is recorded in
[`benchmarks/m4-pro-contact-state-traversal-bypass-2026-09-02.md`](benchmarks/m4-pro-contact-state-traversal-bypass-2026-09-02.md).
Empty resident hit-event scratch bypass is recorded in
[`benchmarks/m4-pro-hit-event-bitset-residency-2026-09-02.md`](benchmarks/m4-pro-hit-event-bitset-residency-2026-09-02.md).
Private shape results and selective synchronization are recorded in
[`benchmarks/m4-pro-private-shape-results-2026-09-02.md`](benchmarks/m4-pro-private-shape-results-2026-09-02.md).
Persistent shape-input reuse is recorded in
[`benchmarks/m4-pro-shape-input-registry-2026-09-02.md`](benchmarks/m4-pro-shape-input-registry-2026-09-02.md).
The complete unconstrained CPU body-finalization traversal bypass and its
loaded-host whole-world signal are recorded in
[`benchmarks/m4-pro-body-finalization-traversal-bypass-2026-09-03.md`](benchmarks/m4-pro-body-finalization-traversal-bypass-2026-09-03.md).
The follow-on removal of the steady awake-body ID scan is recorded in
[`benchmarks/m4-pro-shape-input-order-revision-2026-09-03.md`](benchmarks/m4-pro-shape-input-order-revision-2026-09-03.md).
Private enlarged-proxy residency, direct pair consumption, and the current
whole-world measurements are recorded in
[`benchmarks/m4-pro-resident-pair-moves-2026-09-03.md`](benchmarks/m4-pro-resident-pair-moves-2026-09-03.md).
GPU final pair planning, residual-filter move compaction, and exact topology
differentials are recorded in
[`benchmarks/m4-pro-final-pair-planning-2026-09-03.md`](benchmarks/m4-pro-final-pair-planning-2026-09-03.md).
Exact joint-body pair rejection and articulated mutation differentials are
recorded in
[`benchmarks/m4-pro-joint-pair-registry-2026-09-03.md`](benchmarks/m4-pro-joint-pair-registry-2026-09-03.md).
Fully resident convex-contact state, sim, shape-bound, and body-finalization
ownership is recorded in
[`benchmarks/m4-pro-full-contact-residency-2026-09-03.md`](benchmarks/m4-pro-full-contact-residency-2026-09-03.md).

### Four-point resident contact transport

Convex face contacts may contain up to `B3_MAX_MANIFOLD_POINTS` (four) points.
The resident manifold, preparation, and compact impulse ABIs therefore carry
four points end to end, matching the existing four-point capacity of Box3D's
wide contact solver. Metal preparation, warm-start matching, solve,
restitution, compact store, lazy CPU synchronization, and the
precomputed-manifold bridge all accept the same limit.

This is transport and solver capacity, not a claim that hull-hull narrow phase
is already resident. The Metal manifold generator still emits the documented
sphere, capsule, and hull-sphere subset. Hull-hull SAT, clipping, and feature
generation are the next producer to use the widened ABI; those pairs continue
through the CPU oracle until that differential test gate passes.
