# M4 Pro persistent shape-input registry - 2026-09-02

## Ownership cut

Revision-stable Metal finalization steps reuse the existing 72-byte shape-input
records. The CPU still walks awake body simulations for solver properties, so
that pass now also compares the exact cached body-id sequence. If it matches the
current awake order and resident tree revision, Box3D skips shape counting,
body shape-list traversal, local AABB computation, and geometry/filter/proxy
record writes.

The check uses exact ids rather than a hash. Sleep/wake reorder, topology and
tree revision changes, public transforms, filter-only edits, allocation failure,
or invalid result mappings reject reuse. A rejected registry first synchronizes
stale CPU bounds and then rebuilds from the CPU oracle.

## Correctness and route evidence

- The sparse 1,024-shape resident world records one initial pack and one reuse
  before a public mutation forces the second pack.
- A dedicated 64-shape world records one pack and one reuse, then exact
  sleep/wake swap-removals force two rebuilds. The next unchanged step reuses
  the rebuilt registry.
- Those two awake-order changes explicitly synchronize 64 then 63 stale shape
  records before rebuilding.
- A mask-only filter edit with contact invocation disabled changes no required
  tree topology, but still invalidates the cached no-contact eligibility and
  forces a fourth pack plus the compatibility full-result apply.
- The ten-step convex contact world records one shape-input pack and nine
  reuses while retaining ten compatibility result applies.
- Float and double warning-as-error Metal tests, the full CPU-only suite, full
  AddressSanitizer and float UndefinedBehaviorSanitizer suites, and the double
  UndefinedBehaviorSanitizer Metal suite pass.
- Double precision retains the VF64 far-world containment result with zero
  inward error.

## Performance status

No timing from this checkpoint is accepted because the development host had
multiple high-CPU CoreGraphics workers and other interactive load. A clean-host
whole-world run must report `shapeInputPackCount`, `shapeInputReuseCount`, full
applies, synchronized shapes, and complete `b3World_Step` time.

## Remaining CPU boundary

Cold starts and invalidated registries still count shapes, walk awake body shape
lists, and repack records. The next broad-phase ownership step is GPU-side pair
filtering and deterministic accepted-pair compaction, followed by incremental
contact creation/narrow-phase residency.
