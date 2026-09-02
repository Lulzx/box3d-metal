# Resident contact schedule (2026-09-02)

## Scope

This checkpoint retains the deterministic four-byte-per-lane contact-ID
schedule in its Metal buffer across unchanged solver topology. `b3ConstraintGraph`
now carries a monotonic 64-bit revision advanced at every contact or joint graph
insertion and removal, including swap-removal reorderings.

Schedule reuse requires the same graph revision, SIMD-wide constraint count,
and authoritative resident contact count. Any mismatch performs the original
upstream-color-order pack before command submission. Mixed, callback, recycled,
overflow, and unsupported routes still submit no resident schedule. Restitution
eligibility remains current step data and is not cached with topology.

## Evidence

The 81-contact sphere/capsule differential fixture performs one 336-byte pack
and three resident reuses over four successful steps. It retains the existing
float maximum transform/velocity errors of `5.96e-08`/`3.58e-07`. Adding an
82nd independent touching contact advances the graph revision and produces
exactly the second pack, with no extra reuse; the added body's CPU/GPU velocity
remains within `3.0e-05`.

`b3MetalProfile.contactSchedulePackCount` and
`contactScheduleReuseCount` distinguish topology uploads from unchanged-step
reuse. `lastContactPrepareIndexBytes` remains the logical schedule size rather
than claiming a new upload on reuse.

No loaded-host speedup is claimed. The whole-world resident-contact harness now
prints schedule packs and reuses so a quiet-host run can isolate this ownership
change from compact impulse extraction and the remaining CPU persistence work.
