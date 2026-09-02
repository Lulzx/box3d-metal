# Architecture

## Design objective

Apple Silicon has unified physical memory, but CPU/GPU ownership changes,
command submission, synchronization, cache coherence, and kernel launch latency
still cost time. The backend therefore fuses ordered solver work into one Metal
command buffer per world step and retains body state across substeps.

## World-step command graph

For a supported constrained world, one command buffer performs:

1. Force, damping, gyroscopic velocity integration, and velocity limits.
2. Optional resident convex contact preparation.
3. Optional warm starting for overflow and each active graph color.
4. Biased velocity iterations for joints and contacts.
5. Position and quaternion integration.
6. Unbiased relaxation iterations.
7. Restitution for eligible contacts.
8. Compact resident-convex impulse extraction by contact ID.
9. Optionally, body and awake-shape finalization using the resident solver state.
10. Optional resident tree-leaf updates plus deterministic internal refit.
11. One synchronization followed by compact hit-event exceptions, lazy public-manifold sync, and topology work.

Pair generation is currently a separate experimental command sequence. Metal
retains the existing three dynamic-tree node arrays, counts candidates per moved
proxy, computes a stable hierarchical exclusive scan, and writes candidates in
exact upstream tree and depth-first visitation order. SIMD subgroups scan fixed
256-lane blocks, a short serial kernel prefixes block totals, and a parallel
pass adds block offsets. A monotonic broad-phase revision retains node snapshots
across calls. Supported awake-shape finalization writes enlarged leaf bounds
directly and refits parents in ascending height, preserving Box3D topology and
DFS order without a per-step tree upload. Topology changes and unsupported CPU
mutations invalidate the snapshot. The unchanged CPU callback performs pair-set
deduplication, body/shape/joint/custom filters, compound handling, and contact
creation.

Unconstrained worlds use a smaller fused kernel sequence spanning every
substep. Unsupported constrained worlds keep the reference CPU solver and may
still use the independent Metal position stage if they meet the threshold.

## Data ownership

- `b3BodyState` is copied into a persistent shared Metal buffer once per step.
- Compact body properties exclude collision and finalization fields unused by
  integration kernels.
- Experimental finalization uses separate compact body and shape geometry
  streams so their fields do not widen the default integration record.
- Double builds carry absolute center bits through VF64 exact binary64 addition
  and directed float narrowing; local shape geometry remains float by design.
- CPU convex and mesh contact preparation writes directly into persistent
  shared Metal allocations. Complete resident convex sets instead use a Metal
  preparation kernel. Their post-persistence metadata is retained by contact ID
  and solver submission copies only a four-byte lane schedule.
- Complete resident convex sets extract an 80-byte post-solve record per active
  contact. CPU manifold/event synchronization resolves the manifold by contact
  ID and does not traverse the 1,696-byte SIMD-wide solver records.
- A monotonic constraint-graph revision retains the contact-ID lane schedule
  across unchanged steps. Contact/joint insertion or removal, wide-count
  changes, and authoritative-contact-count changes force an ordered repack.
- Distance and parallel joints are packed into compact type-dense records and
  only their accumulated solver state is unpacked after the command buffer.
- Capacities grow geometrically and buffers are reused.

Every shared C/Metal structure has compile-time size and offset assertions.
Shader-side vectors that would acquire Metal `float3` padding are represented
with scalar fields and explicit helpers.

## Constraint parallelism

Box3D's conflict graph guarantees that constraints in one non-overflow color do
not write the same dynamic body. One GPU thread therefore owns one constraint,
and colors execute in upstream order with buffer barriers between colors.

Convex contacts retain the upstream four-lane wide representation. Mesh
contacts use scalar multi-manifold records. Distance and parallel joints use
separate dense buffers, allowing supported joint types to coexist within a
color without sparse union records.

## Overflow

Overflow constraints can share bodies, so they cannot safely execute in
parallel. A single Metal thread walks each overflow array in deterministic
upstream order. Mixed distance/parallel overflow uses an eight-byte descriptor
per joint containing its type and dense-buffer index. This preserves ordering
with one kernel launch per solver phase instead of one launch per constraint.

## Apple GPU choices

- One ordered command buffer minimizes CPU/GPU round trips.
- Persistent `MTLStorageModeShared` buffers exploit unified memory while keeping
  ownership transitions explicit.
- Threadgroup widths are derived from pipeline execution width and maximum
  occupancy rather than hard-coded.
- The fast path avoids atomics because graph colors establish exclusive writes.
- Overflow stays on the GPU but deliberately sacrifices parallelism for
  correctness.
- Safe Metal math and disabled CPU FP contraction reduce avoidable numerical
  drift against the CPU reference.

## Telemetry

`b3MetalProfile` reports device name, selected threshold, dispatch/fallback
counters for position, unconstrained, contact, joint, finalization, shape, and
pair paths, plus the latest GPU execution time for each category. These
counters are route evidence, not a whole-engine GPU percentage.

## Experimental finalization boundary

`b3World_SetMetalFinalization(world, true)` ports final rotation, body-origin
offset, farthest-point motion and sleep metrics, world-space inverse inertia,
and awake non-CCD shape AABBs. Supported fused solver paths append both kernels
to their existing command buffer. VF64 preserves double-precision outward
rounding on Metal. The 64-byte shape result uses private storage. A bounded
route—continuous collision disabled, no sensors or existing contacts, and all
awake shape contact masks disabled—encodes no full-result blit or CPU apply.
Public AABB queries stage one record on demand; route changes and disable perform
checked bulk synchronization. CCD, events, island sleep mutation, and general
collision consumers remain on the CPU.

## Experimental pair traversal boundary

`b3World_SetMetalBroadPhase(world, true)` ports resident leaf update, internal
refit, traversal, moved-proxy de-duplication, and built-in shape filtering, not
topology ownership or all broad-phase logic. A bounded 64-entry private DFS stack
and average-candidate capacity guard make failure explicit; excessive depth,
candidate volume, allocation, or dispatch failure reruns the complete CPU
traversal. The steady path counts, scans, and writes in one command buffer. If
the exact total exceeds the geometrically retained candidate capacity, the
first call grows the buffer and submits one write-only retry. Supported moving
worlds reuse the resident topology. A revisioned 32-byte-per-shape table rejects
same-body pairs, sensors, and group/category/mask failures during both traversal
passes. Creation, destruction, and filter mutation refresh it; stable steps do
not repack it. A revisioned mirror of Erin's 16-byte open-addressed pair set
uses the same key, hash finalizer, and linear probing to suppress existing
non-compound contacts. Compound parents bypass the lookup because their keys
include child ids. Accepted candidates still return to the CPU.
Each per-move record carries the resident query leaf's shape id and fat
AABB, so CPU filtering no longer dereferences the CPU tree for query metadata.
After a successful device refit, a 256-lane hierarchical scan stably compacts
one 32-byte record per enlarged shape. Proxy bookkeeping consumes only that
deterministic subset, skipping the full-result rescan, enlarged-body bit-set
merge, and second body/shape-list walk. Compatibility routes still stage and
apply the complete result; the bounded resident route defers CPU materialization.
Cold and invalidated shape-registry rebuilds still walk awake body shape lists,
so those transitions are not yet device-resident.

On an unchanged subsequent step, shape finalization reads the previous Metal
fat bounds as its containment oracle. Double builds therefore retain the
conservative VF64 result across steps rather than repacking a rounded CPU
mirror. Reuse requires an identical shape count and broad-phase revision;
buffer growth, topology changes, explicit moves, and rebuilds force a CPU-oracle
reseed. `shapeBoundsResidentDispatchCount` reports successful resident reuse.
`shapeResultApplyCount` and `shapeBoundsSyncCount` distinguish compatibility
applies from selective synchronization.

The 72-byte shape-input allocation is persistent too. On a revision-stable
step, an exact comparison of cached awake body ids preserves every body index
and reuses geometry, filter, proxy, local-bound, and resident-fat-bound records
without counting shapes or walking body shape lists. Sleep/wake swap-removal,
tree revisions, transforms, filter-only changes, invalid mappings, and allocation
failure reject reuse. A rejected registry synchronizes stale CPU mirrors before
repacking from the CPU oracle. `shapeInputPackCount` and
`shapeInputReuseCount` expose the routes.

After existing-contact suppression, ordinary candidates go directly to CPU
joint/custom checks and move-pair append. Compounds retain the complete CPU
child callback, and deterministic contact creation remains CPU-owned.

## Incremental narrow phase

The shape-specialized route batches sphere-sphere, capsule-sphere,
capsule-capsule, and bounded compact hull-sphere local manifold geometry over
the deterministic awake-contact index array. Fixed results carry zero, one, or
two points with exact feature-id ordering. Each record carries explicit
eligibility, so unsupported pairs stay on the ordinary CPU path in the same
step. High-aspect hulls and compact speculative hull-sphere contacts retain CPU
GJK. CPU workers apply Metal geometry through the existing contact update,
preserving allocation, feature-id warm-start matching, material/pre-solve
callbacks, events, recycling, and graph/island transitions.

Double builds carry both absolute position bit patterns and use the vendored
VF64 exact subtraction before narrowing the relative displacement to float at
the same boundary as Box3D's CPU convex collision. Supported sphere, capsule,
and compact hull geometry lives in a persistent registry indexed by Box3D shape
id. The cold path packs primitive endpoints/radii, validates hulls, and
content-deduplicates hull points, planes, and triangulated boundaries into
64-byte shape descriptors; revision-stable dispatches skip shape traversal and
packing. Creation, destruction, and geometry mutation rebuild fail-closed.
Filter mutation conservatively over-invalidates because the geometry and pair
metadata registries currently share a revision.

A separate 64-byte transform record indexed by body id covers static, awake,
and sleeping solver sets without depending on the later solver-state upload.
The cache key combines world step, explicit transform revision, and body-slot
count. Body create/destroy/teleport and replay seek invalidate it; solved motion
refreshes it at the next collision phase. Double records retain all three exact
binary64 position bit patterns for shader-side VF64 subtraction. Unsupported
contact batches return before either registry is built.

Input packing remains; this is not yet resident manifold ownership. The input
is 16 bytes per contact and carries eligibility plus two shape ids. Full
80-byte results are private. A deterministic 256-lane block scan, serial block
prefix, and parallel scatter return only active results in the same command
buffer. Each compact record carries its original contact index and remains
ordered. CPU workers lower-bound once per parallel range and then walk the
compact stream linearly, without a dense lookup allocation. CPU contact
validation and manifold application remain.
The scatter also rotates active normals and frame-A points with the resident
body quaternion. CPU application skips matrix construction and local-to-world
vector transforms. Points remain relative to body A's origin so exact far-world
anchor construction and center-of-mass adjustment retain CPU semantics.

The same scatter writes an identical finalized record to a persistent private
table indexed by Box3D contact id. The 16-byte input's fourth word now carries
that id. Compact output remains ordered by awake-contact input index for the
current CPU application path, while the private copy sets `inputIndex` to the
contact id so its address and identity are independent of input permutation.
The table is exposed only through an explicit diagnostic/fallback blit; normal
steps add no shared stream, command buffer, or wait. Entries are authoritative
only for contacts marked eligible in the current successful dispatch.

That authority now survives into solver setup explicitly. Each collision worker
clears a transient ownership bit before overlap and recycling decisions, then
sets it only after a resident result passes through Box3D persistence, material,
and topology handling. Pre-solve callback contacts remain CPU-owned. Solver
setup counts marked contact ids in graph-color order and exposes SIMD-wide
coverage only when every colored convex contact is resident-table authoritative
and no convex overflow exists.

That gate drives a Metal preparation kernel at the front of the existing solver
command buffer. It reads normal and identity from the private contact-id table
and writes the established 1,696-byte SIMD-wide constraint ABI. CPU-owned
persistence anchors, warm-start impulses, materials, tangent velocity, body
indices, contact generation, point feature IDs, and manifold identity live in a
generation-tagged 152-byte shared table
indexed by contact ID. Collision workers write disjoint records during the
existing persistence pass. Solver submission initializes tail lanes and
bulk-copies each active color's four-byte contact IDs without dereferencing
world contact/manifold storage. The kernel computes Erin's tangent frame,
softness, normal/tangent/twist/rolling
masses, friction centers, lever arms, relative velocities, and projected
warm-start impulses. Mixed, recycled, callback, overflow, stale, or malformed
sets fail closed. If another unsupported constraint rejects the Metal solver
after CPU preparation was skipped, Box3D reruns convex preparation before the
CPU solver fallback. CPU persistence/material table writes, graph scheduling,
events, and topology remain the next residency boundary. After restitution, a
kernel in the same command buffer writes world-axis friction, twist, rolling,
point normal/total impulses, and pre-solve normal velocity into a
generation-tagged 80-byte table indexed by contact ID. Identity, result
generation, contact generation, and point count are validated;
release fallback can still consume the wide records.
The schedule buffer is reused when graph revision and exact wide/contact counts
match. Restitution eligibility remains current step state and is not cached.
The 80-byte result reuses padding for contact generation and two feature IDs.
Before the next fresh supported preparation record is staged, feature-matched
points recover normal impulses from that GPU-authored result; friction, twist,
and rolling terms recover at manifold scope. This makes solver warm starts
independent of freshness in the CPU public-manifold mirror while rejecting
destroyed/recreated contact slots.
Successful resident solves skip the all-contact CPU impulse-store walk. During
the already-required narrow-phase input pack, Box3D retains only IDs whose
shapes requested hit events. One store block processes that compact exception
list, synchronizes qualifying contacts by generation and feature ID, and sets
the existing event bits; final construction therefore keeps contact-ID order.
With no hit requests, the stage touches no contacts. Public contact/body/shape
queries sync requested records, while force debug drawing and snapshots are
explicit boundaries. CPU routes invalidate older GPU result authority before
fallback.
