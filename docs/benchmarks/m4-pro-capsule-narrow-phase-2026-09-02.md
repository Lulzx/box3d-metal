# M4 Pro capsule narrow-phase checkpoint — 2026-09-02

This checkpoint extends the ordered convex-manifold batch from sphere-sphere to
capsule-sphere and capsule-capsule geometry. It remains correctness evidence,
not an accepted whole-world benchmark.

## Geometry and ABI

The fixed result record now carries zero, one, or two local manifold points plus
their exact Box3D feature ids. The Metal kernel ports Erin Catto's point/segment
and segment/segment closest-point routines. Nearly parallel capsule edges use
the same two side-plane clips, endpoint feature pairs, averaged normal, and
two-point ordering as the CPU implementation. Runtime length units feed the
linear slop and speculative distance rather than baking meter-scale constants.

Sphere-sphere, capsule-sphere, and capsule-capsule contacts are eligible only in
the primary ordering selected by Box3D's contact register. Hulls and all complex
shape paths retain explicit per-record CPU fallback. CPU application continues
to own warm-start persistence, materials, callbacks, events, and topology.

## Differential evidence

On Apple M4 Pro, the focused float route covered:

- 32 sphere-sphere contacts;
- eight capsule-sphere contacts;
- one separated capsule-sphere contact with an authoritative zero-point result;
- eight crossed capsule-capsule one-point contacts;
- eight parallel capsule-capsule two-point contacts; and
- one ineligible hull contact in the same batch.

All 57 eligible records matched the CPU point count, normal, points,
separations, feature ids, and order. The maximum direct geometry error and
end-to-end applied-manifold error were both `1.19e-7`. Repeating the unchanged
dispatch produced byte-identical 80-byte result records.

The double/VF64 route covered five eligible records—one of each touching
geometry case plus the separated capsule-sphere case—at a `1e12` world origin,
alongside one ineligible hull. Direct oracle error was `5.96e-8`; applied
manifold error was `6.01e-8`. Exact VF64 subtraction is shared with the sphere
checkpoint.

## Remaining boundary

The kernel still consumes CPU-packed 184-byte contact inputs and exposes a
shared 80-byte result per contact. The next meaningful shape slice is
hull-sphere, followed by resident convex geometry/transform inputs and direct
manifold storage to remove both streams.
