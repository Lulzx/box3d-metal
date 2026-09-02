# Contact-impulse result residency (2026-09-02)

## Scope

After restitution, a kernel in the existing solver command buffer writes one
80-byte record per active resident convex contact into a generation-tagged table
indexed by Box3D contact ID. Records contain world-axis friction, twist and
rolling impulses, plus two points' normal impulse, total normal impulse, and
pre-solve normal velocity.

CPU storage still updates public manifolds and constructs hit events in upstream
graph order. It resolves the manifold by contact ID and consumes the compact
table rather than traversing the complete 1,696-byte SIMD-wide solver records.
Identity, generation, point count, and flags are validated. Unsupported routes
and release-mode validation failure retain the original CPU store path.

This describes the original extraction checkpoint. The lazy-sync follow-on now
bypasses the all-contact CPU store for successful resident solves, synchronizes
public queries on demand, and visits only compact hit-event exceptions.

## Evidence

The 81-contact sphere/capsule differential fixture writes 6,480 compact bytes.
Its 21 SIMD constraints previously exposed 35,616 bytes to CPU impulse storage,
an 81.8% reduction in that input surface. Float maximum transform/velocity error
remains `5.96e-08`/`3.58e-07`; double/VF64 remains
`9.36e-08`/`3.58e-07`.

A separate resident sphere impact matches the CPU hit-event point, normal, and
10 m/s approach speed while reporting one 80-byte result. Portable CPU full,
float and double/VF64 warning-as-error Metal, full AddressSanitizer, full float
UndefinedBehaviorSanitizer, and focused double/VF64 UndefinedBehaviorSanitizer
gates pass.

No loaded-host whole-world speedup is claimed. The resident-contact benchmark
now reports compact and former-wide impulse bytes so the next quiet-host run can
measure whether eliminating the CPU cache walk changes end-to-end time.
Follow-on checkpoints retain contact-ID lanes across unchanged graph revisions,
carry warm starts from resident results, and make public synchronization lazy.
