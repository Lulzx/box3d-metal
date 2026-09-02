# M4 Pro device transform registry — 2026-09-02

## Scope

The resident convex narrow phase previously rebuilt its body-id-indexed
transform registry from CPU body sims whenever the world step advanced. Metal
finalization now writes completed awake-body transforms directly into that
registry in the same command graph as shape AABB generation and tree refit.

The route is bounded to sleeping-disabled, non-CCD steps. A CPU seed remains
necessary for static bodies and after topology or explicit-transform revision
changes. Successful completion publishes the current step/revision only after
the command buffer and contact-prepare status pass. All mutation and fallback
checks therefore fail closed.

Float builds write the world-origin floats used by the narrow phase. VF64
builds additionally construct exact binary64 position bits from the prior
center plus the final delta and rotated local-center offset. Relative collision
coordinates retain the existing exact-subtract-then-narrow boundary.

`b3MetalProfile.narrowPhaseTransformDeviceRefreshCount` counts successful
device publications. The whole-world CSV exposes the same value.

## Structural evidence

On Apple M4 Pro, the 2,048-body unconstrained differential seeded the registry
once and completed one device refresh alongside zero finalization-readback
bytes and one linked-shape traversal bypass.

The supported convex-contact differential completed ten finalization/device
refreshes after one initial CPU registry seed. Every one of its 128 dynamic
body-id records was read directly from the authoritative registry and compared
with the CPU public transform. Maximum transform disagreement was `1.19e-07`
across float and VF64 validation; the broader CPU/Metal solver differential
also remained within its existing transform and velocity tolerances. Reapplying one public transform
immediately made the resident-registry read fail, proving revision invalidation
rather than stale reuse.

No timing claim is attached to this checkpoint because a quiet-host measurement
was not available.

## Remaining boundary

This removes the per-step CPU transform-registry rebuild for the next device
collision consumer. It does not yet keep the solver body-state/property arrays
resident across steps, materialize public transforms lazily, author body move
events on the device, or bypass the contiguous CPU body-finalization walk.
