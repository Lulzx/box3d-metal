# M4 Pro resident hull-geometry checkpoint — 2026-09-02

This checkpoint removes per-contact hull geometry packing from the bounded
hull-sphere Metal route. It is residency, invalidation, and differential
correctness evidence, not an accepted whole-world performance result.

## Ownership cut

Supported compact hulls are validated and content-deduplicated into persistent
Metal buffers. The registry contains point, plane, and triangulated-boundary
streams plus one 32-byte descriptor per Box3D shape slot. A narrow-phase input
is now 184 bytes and carries `shapeIdA`; the MSL kernel resolves the descriptor
and geometry directly instead of receiving six per-contact stream fields and
duplicated hull data.

The registry is keyed to the existing shape-metadata revision. An unchanged
dispatch performs no world-shape traversal or geometry packing. Shape creation,
destruction, and geometry setters invalidate it. Filter changes conservatively
invalidate it too because pair metadata and hull geometry currently share the
revision. Invalid hull metadata, overflow, or allocation failure rejects the
Metal narrow-phase route before partial results are consumed.

## Reuse and invalidation evidence

The focused world contains eight supported hull shapes that share one box hull.
The first direct narrow-phase dispatch records one registry upload. A repeated
direct dispatch and the subsequent world step record at least two reuses, while
the telemetry remains eight supported shapes and one unique hull. Replacing the
first shape with `b3Shape_SetHull` forces exactly the second upload and changes
the registry to eight supported shapes backed by two unique hulls.

The public `b3MetalProfile` exposes `narrowPhaseGeometryUploadCount`,
`narrowPhaseGeometryReuseCount`, `lastNarrowPhaseHullShapeCount`, and
`lastNarrowPhaseUniqueHullCount` so applications and future benchmarks can
distinguish cold rebuilds from stable reuse.

## Differential evidence

On Apple M4 Pro, the float focused batch retains 65 contacts, 62 applied GPU
records, two authoritative separated results, and eight ordered two-point
manifolds. Maximum direct CPU-oracle error remains `1.79e-7`; end-to-end applied
error remains `2.34e-7`; repeated result records remain byte-identical.

The double/VF64 focused batch retains 13 contacts and ten applied GPU records at
a `1e12` world origin. Direct error remains `1.79e-7`, applied error remains
`2.19e-7`, and result records remain byte-identical. High-aspect and positive
speculative hull-sphere cases continue to use the unchanged CPU GJK fallback.

Float and double warning-as-error Metal builds and focused Metal suites pass at
this checkpoint. The full CPU suite, full AddressSanitizer suite with leak
detection disabled for the Metal runtime, full float UndefinedBehaviorSanitizer
suite, and focused double/VF64 UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

The CPU still packs shape endpoints, radii, body rotations, and world
translations into each 184-byte contact input and consumes an ordered 80-byte
result per contact. Manifold allocation, warm-start feature matching, materials,
callbacks, events, contact creation, and topology also remain CPU-owned. The
next narrow-phase residency cut is a persistent transform/primitive table and
direct manifold storage; broader hull coverage still requires the general GJK
simplex and SAT/clipping paths.
