# Resident manifold solver ownership (2026-09-02)

## Scope

This checkpoint carries contact-id-indexed private-manifold authority through
CPU persistence and topology processing into solver setup. It adds no timing or
speedup claim and does not yet replace `b3PrepareContacts_Convex`.

## Contract

- Every collide visit clears the transient Metal-manifold ownership bit.
- Recycling and CPU narrow phase leave the bit clear.
- A successfully consumed resident result sets the bit after `b3UpdateContact`.
- Solver setup counts marked contacts in graph-color order.
- SIMD-wide preparation coverage is non-zero only when every colored convex
  contact is marked; mixed resident/CPU sets fail closed.
- The private manifold table is never staged merely to decide eligibility.

`b3MetalProfile.lastResidentConvexContactCount` reports surviving marked graph
contacts. `lastResidentConvexConstraintCount` reports the complete covered
SIMD-wide range, or zero when any colored convex contact is an exception.

## Evidence

`MetalResidentSolverOwnershipTest` exercises both sides with the same
sphere-sphere contact: recycling disabled yields one resident contact and one
wide constraint; an unchanged step with recycling enabled yields zero for both,
even though narrow phase still produced a private table record.

Validated on the Apple M4 Pro with float and double/VF64 Metal Werror builds,
full CPU, full ASan with leak detection disabled, full float UBSan, and focused
double/VF64 UBSan.
