# M4 Pro finalization shape-traversal bypass — 2026-09-02

## Scope

Metal shape finalization already covered every awake shape, but the CPU body
finalizer still entered each body's linked shape list. On private-result steps
it also recomputed AABBs because the host result pointer was null, even though
the result count and resident tree proved device completion.

The finalizer now treats a nonzero complete Metal shape-result count as the
authority boundary. Non-fast bodies skip their CPU shape list entirely.
Compatibility routes continue with the deterministic flat apply; private
routes retain stale CPU AABB mirrors until the existing checked query/fallback
synchronization runs. Fast/CCD bodies retain Erin's original traversal.

`b3MetalProfile.finalizationShapeTraversalBypassCount` counts successful
phases, and the whole-world CSV exposes the same field.

## Structural evidence

On Apple M4 Pro, the 2,048-body/2,048-shape unconstrained differential reported
one traversal bypass. Before any public bounds query, the tracked CPU shape
AABB remained byte-for-byte equal to its pre-step value while the resident
tree traversal succeeded. The first body AABB query then performed the checked
lazy synchronization, and all CPU/Metal position, rotation, and AABB tolerance
checks passed.

The supported convex-contact differential reported ten traversal bypasses over
ten steps, alongside forty contact dispatches and ten tree refits. Its flat
compatibility apply preserved CPU/Metal transform and velocity agreement.

No whole-world timing claim is attached to this checkpoint because a quiet-host
measurement was not available.

## Remaining boundary

This removes the linked shape traversal and redundant CPU AABB arithmetic, not
the contiguous body finalization walk. The CPU still materializes transforms,
move events, flags, force reset, and previous-pose state from every returned
solver state. Persistent device body authority and deterministic device-authored
move events are required before that final walk can disappear.
