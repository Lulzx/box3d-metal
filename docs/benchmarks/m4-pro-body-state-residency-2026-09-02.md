# M4 Pro cross-step body-state residency — 2026-09-02

## Scope

Metal previously copied the complete CPU awake-state array into its persistent
buffer at the start of every supported solve. Finalization also left integrated
position and rotation deltas in that buffer because the CPU finalizer needed
them to update its body-sim mirror.

On the sleep-disabled, non-CCD route, finalization now publishes the absolute
transform and transient flags, then resets state deltas and transient bits in
place. The CPU bookkeeping pass reconstructs delta motion and sleep metrics
from its prior pose and the device absolute pose. The resident buffer is
therefore already in next-step form after a successful command.

The following step performs an exact byte comparison against the CPU state
mirror. An exact match reuses the resident allocation without a CPU-to-device
copy. Any public velocity/impulse/lock mutation, awake-order change, fallback,
or failed command differs or clears residency and restores the full upload.
This comparison is deliberately transitional: it proves correctness without
requiring every existing public mutator to maintain a new revision counter.

`bodyStateUploadCount`, `bodyStateReuseCount`, and
`lastBodyStateUploadBytes` expose the boundary in the profile and whole-world
CSV.

## Structural evidence

On Apple M4 Pro, the 2,048-body unconstrained one-step differential reported
one initial state upload of 114,688 bytes. Device transform, CPU transform,
rotation, AABB, and VF64 containment checks passed after the new on-device reset
and CPU reconstruction path.

The supported 128-body convex-contact differential ran ten complete steps with
one initial state upload, nine resident reuses, and zero bytes uploaded on its
tenth step. It retained forty contact dispatches, ten private finalizations,
ten device transform refreshes, and CPU/Metal transform and velocity agreement.
After a public linear-velocity mutation, the next step reported a second full
7,168-byte upload while the reuse count remained nine.

The 512-body structural whole-world run covered six solver invocations (five
warmups plus one recorded step) and reported one upload, five reuses, and zero
latest upload bytes. No timing claim is attached because the host was not quiet
enough for a trustworthy comparison.

## Remaining boundary

The state upload is gone on unchanged steps, but the exact comparison still
reads the full CPU array and solved states are still copied back for the CPU
body-finalization walk. Body properties are also repacked each step. The next
boundary is revisioned state/property invalidation plus device-authored move
events and lazy CPU body mirrors; only then can the final body traversal and
state readback disappear.
