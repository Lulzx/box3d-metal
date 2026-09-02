# Resident contact schedule (2026-09-02)

The deterministic four-byte-per-lane contact-ID schedule now remains in its
Metal buffer across unchanged solver topology. `b3ConstraintGraph` has a
monotonic revision advanced at contact or joint insertion/removal, including
swap-removal reorderings.

Reuse requires the same graph revision, SIMD-wide constraint count, and
authoritative resident-contact count. Any mismatch repacks in upstream color
order. Restitution eligibility remains current step data and is not cached.

The 81-contact differential fixture performs one 336-byte pack and three reuses
over four stable steps, retaining `5.96e-08` transform and `3.58e-07` velocity
maximum error. Adding an 82nd touching contact forces exactly the second pack;
the new body's CPU/GPU velocity remains within `3.0e-05`.

The 512-contact loaded-host smoke recorded one pack and nine reuses over ten
dispatches. No wall-clock speedup is claimed from that run.
