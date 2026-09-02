# M4 Pro finalization readback bypass — 2026-09-02

## Scope

The private body-finalization result already feeds Metal shape AABB generation
and resident tree refit. Stable worlds with sleeping and continuous collision
disabled now omit the separate 100-byte-per-body host blit. The CPU oracle pass
recomputes finalization arithmetic from the returned solver states, so public
transforms, move events, flags, and force reset keep their upstream behavior.

Sleeping or CCD-capable worlds retain the checked result readback. Allocation,
encoding, or solver fallback also remains fail-closed. The profile separates
the routes with `lastFinalizationReadbackBytes` and
`finalizationReadbackBypassCount`.

## Structural evidence

On Apple M4 Pro, the 2,048-body unconstrained integration differential reported
one finalization dispatch, one bypass, and zero finalization readback bytes. Its
private result still produced 2,048 shape results and fed resident tree refit.
CPU/Metal position, rotation, and AABB errors remained within the existing
tolerances.

The supported convex-contact differential ran ten complete steps and reported
ten finalization bypasses with zero finalization readback bytes. Forty colored
contact dispatches and ten resident tree refits completed; CPU/Metal transform
and velocity comparisons passed.

The sleep-enabled control reported one finalization dispatch, zero bypasses,
and one 100-byte result readback. This proves the required compatibility path
was retained rather than silently disabled.

No timing comparison is recorded for this checkpoint because the host was not
quiet enough for a trustworthy whole-world measurement.

## Remaining boundary

The 100-byte-per-body finalization stream is gone on the bounded route, but the
solver body states still return to shared memory and the CPU still walks every
body to materialize transforms and events. The next meaningful port is
device-authoritative body transforms plus deterministic compact move events,
with lazy fail-closed CPU synchronization for queries and fallback.

The subsequent
[`shape traversal bypass`](m4-pro-finalization-shape-traversal-bypass-2026-09-02.md)
removes the linked per-body shape walk while leaving this contiguous body
bookkeeping boundary explicit.
