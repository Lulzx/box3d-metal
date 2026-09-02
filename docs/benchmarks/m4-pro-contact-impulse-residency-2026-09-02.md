# Contact-impulse result residency (2026-09-02)

## Scope

This checkpoint moves post-solve impulse extraction onto Metal for complete
resident convex contact sets. A kernel runs after restitution in the existing
solver command buffer and writes one 80-byte record per active contact into a
shared, generation-tagged table indexed by Box3D contact ID.

The record contains world-axis friction impulse, twist and rolling impulses,
and two points' normal impulse, total normal impulse, and pre-solve normal
velocity. CPU storage still walks contacts in upstream graph order to update
public manifolds and construct hit events. It now reads compact contact records
instead of rereading the complete 1,696-byte SIMD-wide solver constraints.
Generation, contact ID, point count, and flags are validated before a record is
used; release builds retain the original wide-record path as a fail-closed
fallback. Mixed, recycled, pre-solve callback, overflow, and unsupported worlds
remain on the CPU path.

## Correctness evidence

The 81-contact sphere/capsule differential fixture produced 81 compact records,
or 6,480 bytes. Its 21 SIMD constraints previously exposed a 35,616-byte
post-solve CPU read stream. This is an 81.8% reduction in the impulse-storage
input surface, not elimination of CPU public-manifold synchronization.

Four consecutive four-substep steps retained maximum float transform error
`5.96e-08` and velocity error `3.58e-07` against the CPU oracle. The validation
build asserts that every resident graph lane resolves to a current compact
record. A separate hit-event fixture matched the CPU event point, normal, and
10 m/s approach speed while reporting one 80-byte result.

`b3MetalProfile.lastContactImpulseResultBytes` reports active records authored
by the latest successful resident solve. Recycling and callback exception routes
report zero. The existing CPU solver fallback remains authoritative if the
Metal command buffer or preparation validation fails.

## Performance boundary

No quiet-host whole-world timing is claimed here. The
`metal_resident_contact_benchmark` now reports both compact impulse bytes and the
equivalent former wide-record traversal, alongside its contact-ID preparation
schedule. A future measurement must determine whether the smaller CPU cache
stream changes end-to-end time. The next ownership reduction is to retain the
graph schedule and make public-manifold/event synchronization lazy or exception
only; topology and event ordering remain CPU responsibilities at this checkpoint.
