# M4 Pro resident body-transform checkpoint — 2026-09-02

This checkpoint removes repeated body rotations and translations from the
ordered narrow-phase input. It is residency, invalidation, and differential
correctness evidence, not an accepted whole-world performance result.

## Ownership cut

A persistent 64-byte record indexed by Box3D body id stores rotation, narrowed
float world position, exact binary64 world-position bits, and validity. The
shape registry carries each shape's body id, so the MSL kernel follows two shape
ids to both shape geometry and body transforms. The per-contact input falls from
120 bytes to 16 bytes: eligibility, `shapeIdA`, `shapeIdB`, and padding.

The transform registry covers static, awake, and sleeping body simulation sets;
it does not depend on the solver state buffer, which is populated after the
collision phase. Its key combines `world->stepIndex`, an explicit body-transform
revision, and the body-slot count. Same-step direct dispatches reuse it. Body
creation, destruction, and `b3Body_SetTransform` advance the revision. A solved
step advances `stepIndex`, forcing fresh transforms at the next collision.
Replay seek advances revisions so an older restored step cannot match stale
buffers by coincidence. Unsupported batches return before registry work.

## Route evidence

The focused mixed batch records one cold transform upload and two same-step
reuses. Hull and sphere geometry mutations rebuild only the shape registry.
`b3Body_SetTransform` causes exactly the second transform upload while the shape
upload count remains unchanged. Three subsequent solved steps produce two more
uploads at the next collision boundaries, for four total, without rebuilding
shape geometry.

The public profile exposes `narrowPhaseTransformUploadCount` and
`narrowPhaseTransformReuseCount` alongside the geometry counters.

## Differential evidence

The float batch remains 65 contacts with 62 applied GPU records. Direct
CPU-oracle error remains `1.79e-7`, applied error remains `2.34e-7`, and repeated
records remain byte-identical. The double/VF64 batch remains 13 contacts with
ten applied GPU records at a `1e12` origin; direct error remains `1.79e-7` and
applied error remains `2.19e-7`. VF64 subtraction now reads exact bits from the
body registry rather than each contact record.

Float and double warning-as-error focused Metal suites pass. The full CPU suite,
full AddressSanitizer suite with Metal-runtime leak detection disabled, full
float UndefinedBehaviorSanitizer suite, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

The CPU still validates and writes one 16-byte record per awake contact, waits
for the command buffer, and consumes an ordered 80-byte result per contact.
Manifold allocation, persistence, callbacks, events, contact creation, and
topology remain CPU-owned. Direct manifold storage or active-result compaction
is the next ownership cut.
