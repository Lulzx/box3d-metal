# M4 Pro private VF64 move events — 2026-09-02

## Scope

The bounded sleep-disabled, non-CCD finalization route now emits public body
move data directly from the finalization kernel. Records remain in private
Metal storage in awake-sim order. The CPU finalization walk still performs
engine bookkeeping, but it no longer writes the public move-event array.

Each 72-byte record snapshots rotation, body identity, user data, a float
compatibility position, and exact binary64 position bits. In double-precision
builds those bits are the result of VF64 correctly rounded additions from
commit `729021777455da72db8809d9ef1269c677d88b3f`. This extends the existing
VF64 AABB and relative-contact boundary through the public move-event API.

`b3World_GetBodyEvents` performs the explicit private-to-shared blit and public
ABI conversion. If the application never asks for body events before the next
step, the old private stream is simply replaced and zero event bytes cross the
boundary.

## Structural evidence

The 2,048-body differential verifies that the event stream remains private
after the step: one device dispatch, one CPU event-write bypass, zero syncs,
and zero readback bytes. The first public event query copies 147,456 bytes,
returns 2,048 records in the same order as the CPU oracle, preserves user data
and body identity, and reports `fellAsleep == false`. Every event translation
is bit-identical to the device-authoritative public body transform. Float and
double/VF64 focused Metal suites pass; the VF64 far-world AABB containment
underflow remains zero.

A forced-sleep transition before the public query also passes: the transition
materializes once, finds the snapshotted body identity, and marks the event's
`fellAsleep` flag through Erin's unchanged sleep path.

The ten-step 128-body resident-contact differential leaves events unobserved
and reports ten device dispatches, ten CPU-write bypasses, zero syncs, and zero
latest readback bytes.

A 512-body whole-world structural run covered five warmups plus one recorded
step. It reported six private finalization/readback bypasses, six shape-walk
bypasses, six move-event dispatches, zero event syncs, zero latest event bytes,
one body-state upload, and five resident state reuses.

No timing claim is attached to this checkpoint because the host was not quiet
enough for a trustworthy comparison.

## Remaining boundary

This removes event construction from the CPU walk, not the walk itself. The
CPU mirror still consumes resident transforms, reconstructs sleep metrics,
resets forces and transient flags, and receives solved body states. Revisioned
body-property/state authority and lazy public body mirrors remain necessary
before that traversal and state readback can disappear.
