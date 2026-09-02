# M4 Pro hull-sphere narrow-phase checkpoint — 2026-09-02

This checkpoint extends the ordered convex-manifold batch to bounded compact
hull-sphere geometry. It is correctness and fallback evidence, not an accepted
whole-world benchmark.

## Geometry and boundary

The Metal kernel transforms the sphere center into the hull frame, finds the
best face plane, and searches a CPU-triangulated hull boundary for the closest
face, edge, or vertex point. A best-face projection avoids poorly conditioned
large-triangle barycentrics for ordinary face contacts. Deep overlap uses
Box3D's best-plane normal and separation rule.

The route is deliberately bounded. Hulls must have a valid three-dimensional
extent and a maximum-to-minimum local extent ratio no greater than 16. Compact
contacts with positive separation inside Box3D's speculative distance return
per-record ineligible results and run CPU GJK. High-aspect hulls do the same.
Clearly separated compact pairs return an authoritative zero-point GPU result.
This preserves the existing large-ground restitution oracle instead of hiding
an accumulated trajectory regression behind a wider tolerance.

Each 216-byte contact input references packed point, plane, and triangle
streams; the 80-byte result remains ordered with the awake contact array. Hull
geometry is currently duplicated per contact. CPU application clears the GJK
simplex cache to a valid cold state for applied hull results, while retaining
manifold persistence, callbacks, events, and topology ownership.

## Differential evidence

On Apple M4 Pro, the focused float batch had 65 contacts and 62 applied GPU
records. It covered face, edge, corner, deep-overlap, speculative-fallback, and
authoritative-separated compact hull-sphere cases, plus a high-aspect
hull-sphere fallback and a hull-hull fallback. Existing sphere/capsule coverage
remained in the same batch, including eight ordered two-point capsule
manifolds. Maximum direct CPU-oracle error was `1.79e-7`; end-to-end applied
manifold error was `2.34e-7`. Repeating the unchanged dispatch produced
byte-identical result records.

The double/VF64 batch had 13 contacts and ten applied GPU records at a `1e12`
world origin. Direct error was `1.79e-7`; applied error was `2.19e-7`. The
existing 64-sphere restitution world uses a high-aspect ground and therefore
stayed on CPU GJK, preserving its `1.19e-7` transform and `2.38e-7` velocity
agreement.

## Remaining boundary

Exact speculative and high-aspect hull-sphere behavior still needs the general
GJK simplex path. Hull-capsule and hull-hull need their SAT/clipping paths.
Before broadening those routes, hull geometry and transforms should become a
deduplicated persistent registry and manifolds should be written into resident
storage so the new geometry streams do not become another CPU packing tax.
