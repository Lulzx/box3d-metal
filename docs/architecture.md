# Architecture

## Design objective

Apple Silicon has unified physical memory, but CPU/GPU ownership changes,
command submission, synchronization, cache coherence, and kernel launch latency
still cost time. The backend therefore fuses ordered solver work into one Metal
command buffer per world step and retains body state across substeps.

## World-step command graph

For a supported constrained world, one command buffer performs:

1. Force, damping, gyroscopic velocity integration, and velocity limits.
2. Optional warm starting for overflow and each active graph color.
3. Biased velocity iterations for joints and contacts.
4. Position and quaternion integration.
5. Unbiased relaxation iterations.
6. Restitution for eligible contacts.
7. Optionally, body and awake-shape finalization using the resident solver state.
8. Optional resident tree-leaf updates plus deterministic internal refit.
9. One synchronization followed by CPU impulse storage and topology work.

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
- Convex and mesh contact preparation writes directly into persistent shared
  Metal allocations; no redundant constraint upload/readback is performed.
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
the same boundary as Box3D's CPU convex collision. Input packing and a shared
geometry result array remain; this is not yet resident manifold ownership. The
current input and result records are 216 and 80 bytes per contact respectively,
and hull point/plane/triangle geometry is still duplicated into side streams.
