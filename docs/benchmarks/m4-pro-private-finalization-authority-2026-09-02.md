# M4 Pro private finalization authority — 2026-09-02

## Scope

The body-finalization kernel previously wrote its 100-byte-per-body result into
one shared Metal allocation. Both GPU shape/AABB work and the CPU body apply
walk consumed that allocation, so device and host authority could not be
separated.

Finalization results now live in private device storage. Shape finalization and
resident tree refit read that private buffer in the same command graph. A
separate shared mirror is filled by an explicit checked blit only for the
remaining CPU apply pass. Allocation or encoder failure returns through the
existing CPU fallback boundary.

This does not yet remove the transfer: it makes the transfer optional in the
architecture. `b3MetalProfile.lastFinalizationReadbackBytes` reports the exact
remaining mirror size rather than hiding it inside a shared allocation.

## Evidence

The 2,048-body integrated world reported one finalization dispatch and 204,800
readback bytes. Its private result fed 2,048 shape results and a resident tree
refit before the CPU mirror was applied. CPU/Metal position, rotation, and AABB
errors remained within the existing tolerances.

The direct 4,096-body finalization oracle also passed through the new blit
mirror. The resident shape compaction, broad-phase traversal, contact, joint,
fallback, and VF64 paths remained covered by the Metal differential suite.

## Next boundary

Device-authoritative body transforms and GPU-authored move-event records can now
be added without moving AABB or broad-phase consumers back to shared storage.
Once their lazy CPU synchronization is fail-closed, the finalization blit can be
omitted and `lastFinalizationReadbackBytes` can reach zero on stable worlds.
