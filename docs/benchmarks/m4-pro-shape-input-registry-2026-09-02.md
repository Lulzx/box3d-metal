# M4 Pro persistent shape-input registry - 2026-09-02

## Ownership cut

Revision-stable Metal finalization steps reuse the existing 72-byte shape-input
records. The CPU still walks awake body simulations for solver properties, so
that pass also compares the exact cached body-id sequence. A match skips shape
counting, body shape-list traversal, local AABB computation, and
geometry/filter/proxy record writes.

Sleep/wake reorder, topology and tree revision changes, public transforms,
filter-only edits, allocation failure, or invalid result mappings reject reuse.
A rejected registry first synchronizes stale CPU bounds and then rebuilds from
the CPU oracle.

## Correctness and route evidence

- The sparse 1,024-shape world records one pack and one reuse before a transform
  forces its second pack.
- A dedicated 64-shape world records one pack and one reuse; sleep/wake
  swap-removals then force two rebuilds and synchronize 64 then 63 records.
- A mask-only filter edit with contact invocation disabled still invalidates the
  cached eligibility bit and forces a fourth pack plus compatibility apply.
- The ten-step convex contact world records one pack and nine reuses while
  retaining ten compatibility result applies.
- Float/double warning-as-error Metal, full CPU-only, full AddressSanitizer and
  float UndefinedBehaviorSanitizer, and double UndefinedBehaviorSanitizer Metal
  gates pass.
- VF64 far-world containment retains zero inward error.

## Performance status

No timing is accepted because the host had multiple high-CPU CoreGraphics
workers and other interactive load. A clean-host whole-world run must report
input packs/reuses, full applies, synchronized shapes, and complete step time.

## Next boundary

Cold and invalidated registries still repack on the CPU. The next broad-phase
ownership step is GPU-side pair filtering and deterministic accepted-pair
compaction, followed by incremental contact creation/narrow-phase residency.
