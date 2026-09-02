# Resident convex contact preparation (2026-09-02)

## Scope

This checkpoint ports Erin Catto's colored convex contact-preparation arithmetic
to Metal when the entire colored convex set is backed by the current private
contact-id manifold table. It adds no whole-world speedup claim.

The kernel is encoded before warm start in the existing contact-solver command
buffer. It writes the established 1,696-byte `b3ContactConstraintWide` ABI, so
the warm-start, solve, restitution, and impulse-store kernels are unchanged.
Normal and contact identity come directly from the private resident table.

## Remaining stream and fallback contract

The CPU still walks contacts in graph-color order and packs 144 bytes per contact
lane: body indices, persistent anchors and separations, warm-start impulses,
materials, tangent velocity, and manifold storage identity. Removing that walk
requires a resident persistence/material exception pipeline; it is not claimed
by this checkpoint.

The route is fail closed:

- every colored convex contact must have current resident ownership;
- convex overflow and pre-solve callback contacts stay on CPU preparation;
- recycled, mixed, stale, or malformed records cannot authorize the kernel;
- if another constraint type rejects the Metal solver after CPU preparation was
  skipped, Box3D reruns convex preparation before the CPU solver fallback.

`contactPrepareDispatchCount` counts successful world-step preparations.
`contactPrepareFallbackCount` counts prepare-on-fallback recovery, not ordinary
ineligible CPU preparation.

## Correctness evidence

`MetalResidentContactPrepareDifferentialTest` runs 81 independent contacts over
four four-substep world steps: 64 sphere pairs plus 17 parallel capsule pairs,
covering multiple SIMD lanes, a tail lane, two-point manifolds, friction,
rolling resistance, tangent velocity, restitution, and warm starting. The float
build reported maximum transform error `5.96e-08` and velocity error `3.58e-07`;
the double/VF64 build reported `9.36e-08` and `3.58e-07`.

`MetalContactPrepareFallbackTest` combines one resident contact with an
unsupported revolute joint and observes one CPU preparation recovery with CPU
oracle velocity agreement. `MetalContactPreparePreSolveExceptionTest` verifies
that a called pre-solve callback prevents resident preparation authority.

The portable CPU suite, float and double/VF64 Metal warning-as-error suites,
full AddressSanitizer and float UndefinedBehaviorSanitizer suites, and focused
double/VF64 UndefinedBehaviorSanitizer Metal suite passed on the Apple M4 Pro.
Timing was deliberately omitted because the host was not quiet enough for a
defensible end-to-end comparison.
