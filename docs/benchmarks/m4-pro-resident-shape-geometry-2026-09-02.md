# M4 Pro resident shape-geometry checkpoint — 2026-09-02

This checkpoint generalizes the persistent hull registry to the full supported
narrow-phase shape surface. It is residency and differential correctness
evidence, not a loaded-host performance claim.

## Ownership cut

Each live sphere, capsule, or supported compact hull has a 64-byte record indexed
by Box3D shape id. Sphere and capsule records retain both local endpoints and
their radius. Hull records retain offsets into the content-deduplicated point,
plane, and triangulated-boundary streams. The MSL kernel loads both shape
records directly.

Removing four per-contact primitive `float4` groups shrinks the ordered input
from 184 to 120 bytes. The CPU no longer writes shape types, endpoints, radii,
or hull descriptors for each contact. It still writes two body rotations and
world translations, including exact binary64 position bits for VF64 builds.

## Reuse and invalidation evidence

The prior cold/reuse evidence remains: one upload, at least two stable reuses,
and eight hull shapes deduplicated to one unique hull. Replacing one hull causes
exactly the second upload and two unique hulls. Replacing the first sphere with
`b3Shape_SetSphere` causes exactly the third upload, proving that primitive
geometry mutation also rebuilds before the next dispatch.

## Differential evidence

The float batch remains 65 contacts with 62 applied GPU records, two
authoritative separated records, and eight ordered two-point manifolds. Direct
CPU-oracle error remains `1.79e-7`, applied error remains `2.34e-7`, and repeated
records remain byte-identical.

The double/VF64 batch remains 13 contacts with ten applied GPU records at a
`1e12` origin. Direct error remains `1.79e-7`, applied error remains `2.19e-7`,
and exact VF64 subtraction remains the world-translation boundary.

Float and double warning-as-error focused Metal suites pass. The full CPU suite,
full AddressSanitizer suite with Metal-runtime leak detection disabled, full
float UndefinedBehaviorSanitizer suite, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

The 120-byte stream still repeats body rotations and translations for every
contact, and the CPU still consumes an ordered 80-byte result. A body-id-indexed
transform registry must cover awake and static bodies without relying on the
solver state buffer, which is populated after collision. Direct manifold
storage remains the following cut.

The follow-on
[`resident body-transform checkpoint`](m4-pro-resident-body-transforms-2026-09-02.md)
retains static and awake transforms by body id and reduces the contact input to
16 bytes. The ordered manifold-result stream remains.
