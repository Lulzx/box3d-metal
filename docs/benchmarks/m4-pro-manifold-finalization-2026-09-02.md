# M4 Pro manifold-finalization checkpoint — 2026-09-02

This checkpoint moves the first persistent-manifold finalization arithmetic
off CPU workers. It is ownership and differential correctness evidence, not an
accepted whole-world performance result.

## Ownership cut

The active-result scatter now uses the resident body-A quaternion to rotate the
local contact normal and each frame-A point into world axes before writing the
compact shared record. This is fused into the existing compaction scatter: it
adds no kernel, command buffer, wait, or intermediate shared stream.

Points remain relative to body A's origin. The CPU therefore retains exact
large-world body-origin subtraction, center-of-mass adjustment, manifold
allocation, feature-id persistence, materials, pre-solve callbacks, events,
and graph/island transitions. Unsupported records continue through the
unmodified scalar collision and finalization path.

## Differential evidence

The float mixed batch still compacts 65 inputs to 62 strictly ordered records.
The records are byte-identical across repeated dispatches, direct world-oriented
geometry error is `1.79e-7`, and end-to-end applied error is `2.33e-7`.

The double/VF64 batch still compacts 13 inputs to ten records at a `1e12`
origin. Exact VF64 relative translation remains before float convex collision;
only the local-vector quaternion rotation moves to the GPU. Direct error remains
`1.79e-7` and applied error is `1.94e-7`.

Float and double warning-as-error focused Metal suites pass. The full CPU suite,
full AddressSanitizer suite with Metal-runtime leak detection disabled, full
float UndefinedBehaviorSanitizer suite, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

The CPU still walks every active compact record and materializes persistent
`b3Manifold` objects. The next cut is a stable contact-id-indexed manifold table
that can feed Metal contact preparation directly while exposing only callback,
event, topology-transition, and unsupported-geometry exceptions to the CPU.
