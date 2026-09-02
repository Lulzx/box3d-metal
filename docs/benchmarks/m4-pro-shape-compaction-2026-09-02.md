# M4 Pro enlarged-shape compaction - 2026-09-02

## Scope

This checkpoint adds a deterministic GPU compaction stage after awake-shape
finalization. A 256-lane scan computes stable per-block offsets, a serial block
prefix preserves global order, and a parallel scatter writes one 32-byte record
per enlarged shape. Successful resident refits use this selective stream for
CPU proxy bookkeeping instead of rescanning every 64-byte shape result and
walking enlarged bodies' shape lists.

The full 64-byte result remains CPU-visible and is still consumed in parallel
to keep `b3Shape_GetAABB`, CPU queries, CCD, topology mutation, and fallback
state current. This is therefore an intermediate ownership cut, not evidence
that shape bounds are fully device-authoritative.

The follow-on resident-bounds change makes the previous Metal fat bounds the
containment oracle on an unchanged second step. Float builds reuse the prior
Metal result; double builds reuse the conservative bound produced by
VF64-metal's exact additions and directed narrowing. Shape count and the
monotonic broad-phase revision guard reuse, so external CPU mutations reseed
from the CPU state.

## Correctness evidence

- A sparse 1,024-shape world alternates moving and stationary awake bodies.
  Metal reports exactly 512 compacted records across four scan blocks.
- Every compacted proxy matches upstream deterministic shape order.
- The compact count exactly matches the CPU move array produced from it.
- The sparse world deliberately poisons all 512 moving shapes' CPU fat-AABB
  mirrors after its first step. Its second step still emits exactly the same
  512 records from resident bounds. A subsequent public body transform changes
  the tree revision and demonstrably prevents resident reuse.
- A 2,048-shape mixed sphere/capsule/hull world compacts all 2,048 moving
  shapes, reuses the resident tree, and preserves exact raw pair traversal.
- Float, double-precision VF64, warning-as-error, ASAN, UBSAN, and CPU-only
  suites pass.

## Performance status

No whole-world timing is accepted for this checkpoint. During validation the
host load average remained above 5, with WindowServer and Chrome processes each
using substantial CPU. Timing under that load would not satisfy the repository
clean-host evidence gate. The next accepted comparison must measure complete
`b3World_Step` time against the preceding resident-refit checkpoint in separate
warm processes and report the compacted fraction for each workload.
