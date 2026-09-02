# M4 Pro sphere narrow-phase checkpoint — 2026-09-02

This checkpoint moves the first shape-specialized narrow-phase route to Metal:
sphere-sphere local manifold geometry. It is an architectural and correctness
checkpoint, not accepted whole-world performance evidence.

## Ownership boundary

One command buffer covers the contact-index array in deterministic order.
Eligible sphere records return local normal, midpoint, separation, and touching
state. Ineligible capsules, hulls, meshes, height fields, and compounds remain
on the CPU in the same batch.

The CPU deliberately retains contact recycling, manifold allocation, feature-id
warm-start matching, material callbacks, pre-solve callbacks, hit/contact events,
and graph/island mutation. This keeps Erin Catto's state transition order as the
oracle while geometry routes move incrementally.

Double-precision worlds carry both world-position bit patterns to Metal. The
shader uses the vendored VF64 exact IEEE-754 subtraction from commit
`729021777455da72db8809d9ef1269c677d88b3f`, then rounds the relative displacement
to float at the same boundary as `b3InvMulWorldTransforms`.

## Differential evidence

On Apple M4 Pro:

- Float: 32 eligible sphere contacts plus one unsupported hull contact; maximum
  direct CPU-oracle error and end-to-end applied-manifold error were both
  `1.19e-7`.
- Double: one sphere contact at a `1e12` world origin plus one unsupported hull
  contact; VF64 maximum direct error was `5.96e-8` and applied-manifold error was
  `5.98e-8`.
- Repeating the same dispatch produced byte-identical result records.
- The unsupported hull record remained ineligible and was updated by the normal
  CPU path in the same world step.

Focused float and double warning-as-error Metal suites passed. CPU-only, ASAN,
and UBSAN coverage must also pass before publication.

## Timing boundary

The printed microsecond-scale GPU times are route diagnostics from a host that
was not established as benchmark-clean. They are not speedup evidence. This
route still packs contact inputs on the CPU and exposes a shared geometry result
array, so it does not yet prove a device-resident manifold pipeline.

Capsule-sphere and capsule-capsule geometry followed in the capsule checkpoint.
Resident convex shape/transform inputs and direct manifold storage are still
needed before the shared narrow-phase streams can disappear.
