# Resident contact finalization (2026-09-02)

## Scope

This checkpoint moves the common convex contact's per-point persistence and
finalization arithmetic into the existing Metal manifold scatter. The kernel
validates contact ID, contact generation, and prior result generation; claims
matching feature IDs in upstream order; and emits the matching normal impulse
and persisted bit. The CPU application path no longer copies the old manifold
or scans its points for resident results.

The same pass now emits world-oriented anchors relative to both bodies' centers
of mass. The body registry retains local centers, and double builds use the
existing VF64 exact world-position subtraction before narrowing to Box3D's
float anchor boundary. This removes CPU local-to-world and origin-to-COM anchor
work for supported contacts.

The revisioned shape registry now carries the first convex material. Metal
computes Box3D's default geometric-mean friction, maximum restitution, scaled
rolling resistance, and rotated relative tangent velocity. A distinct material
revision rebuilds this registry after public material mutation. Arbitrary user
friction and restitution callbacks remain explicit CPU exceptions; rolling and
tangent finalization can still come from Metal for those contacts. Pre-solve
callbacks, manifold allocation, recycling, events, and graph/island topology
remain CPU-owned.

The compact shared result grows from 80 to 160 bytes and the shape descriptor
from 64 to 96 bytes. This is an intermediate ownership checkpoint, not the
elimination of the compact result stream. The next structural step is to compact
only callback/topology exceptions and consume unchanged resident contacts by ID.

## Evidence

Float and double/VF64 Metal `-Werror` builds pass the complete `MetalTest`
group. The convex-manifold differential validates both COM-relative anchors,
feature IDs, separations, default material values, deterministic compact order,
and resident-table identity. Float reports `1.79e-07` maximum oracle error and
`2.33e-07` application error; double/VF64 reports `1.79e-07` and `1.94e-07`.

The warm-start fixture poisons the CPU normal impulse before the next step and
then requires the applied point to be persisted with the exact prior resident
impulse. Telemetry reports one GPU persistence match and the CPU/GPU velocity
error remains `4.47e-08`.

A separate callback fixture verifies custom friction `0.123` and restitution
`0.456` are produced by CPU callbacks while Metal supplies the expected rolling
resistance and tangent velocity. Material mutation forces exactly one geometry
registry rebuild. No wall-clock speedup is claimed for this checkpoint.

A loaded-host 512-contact smoke (eight warmups, two measured steps) reported ten
preparation dispatches, one schedule pack, nine reuses, ten impulse-store
bypasses, and zero event/public synchronizations. Wall-clock was `0.188583 ms`
CPU versus `1.436687 ms` Metal (`0.131x`); this is retained only as a regression
smoke and is not quiet-host performance evidence.
