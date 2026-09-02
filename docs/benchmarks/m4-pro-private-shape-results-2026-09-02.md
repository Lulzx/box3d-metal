# M4 Pro private shape results and selective synchronization - 2026-09-02

## Ownership cut

The full 64-byte awake-shape result moved from shared to private Metal storage.
The stable 32-byte enlarged subset remains shared because CPU proxy/contact
bookkeeping still consumes it. A full blit is encoded only for compatibility
routes that still run the CPU shape-result apply task.

The no-full-readback route is deliberately bounded to worlds with continuous
collision disabled, no sensors or existing contacts, and contact masks disabled
on every awake shape. On a successful resident refit it performs zero full
shape-result applies. Public shape/body AABB queries blit one requested record;
route changes, mutation invalidation, and Metal disable use checked bulk
synchronization. Any synchronization failure keeps the previous safe route or
skips the step instead of exposing stale CPU bounds.

## Correctness evidence

- The 1,024-shape sparse test performs three resident shape dispatches with
  `shapeResultApplyCount == 0`.
- A deliberately corrupted CPU AABB is repaired by one 64-byte public-query
  synchronization.
- A public body transform invalidates its old result index; the next step bulk
  synchronization skips both that CPU-authoritative shape and the already
  synchronized query shape.
- Disabling Metal synchronizes the new result generation and leaves no stale
  CPU-bound state.
- The 2,048-shape mixed world performs zero full applies. Its body query plus
  shape queries synchronize exactly 2,048 records and preserve the existing
  CPU/Metal AABB tolerance.
- The ten-step convex contact world remains on the compatibility route and
  records exactly ten full applies.
- Double precision retains the VF64 far-world containment result with zero
  inward error.

## Remaining traversal

This checkpoint removes the shared full-result stream and flat CPU apply only
for the bounded route. `b3MetalPackShapeInputs` still counts shapes and walks
awake body shape lists each step to pack geometry, ids, filters, and proxy keys.
The next structural step is a persistent shape registry updated by topology
mutators so normal steps dispatch resident input records without that traversal.

## Performance status

No timing from the development host is accepted for this checkpoint. An
accepted whole-world comparison must use separate warm processes on a clean
host and report full applies, explicitly synchronized shape count, compacted
shape count, and complete `b3World_Step` time.
