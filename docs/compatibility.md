# Compatibility contract

## Reference

The behavioral reference is unmodified Box3D commit
`47d7f7cc7e091142c08d11dc7d2e493c5d34f536`. Metal acceleration is opt-in per
world. The public body, shape, joint, event, recording, and query APIs remain
the Box3D APIs from that baseline plus the small Metal control/profile API.

Metal results are tolerance-equivalent to the CPU path. Bit-identical
cross-platform determinism is not promised because CPU and GPU floating-point
evaluation order differs.

## Supported GPU-resident solver surface

| Surface | Metal behavior |
| --- | --- |
| Unconstrained velocity and position integration | Fused across all substeps |
| Colored convex contacts | Normal, friction, tangent velocity, twist friction, rolling resistance, restitution |
| Scalar mesh contacts | Multi-manifold solve, friction, rolling resistance, restitution |
| Distance joints | Rigid, spring, lower/upper limit, motor, static/dynamic bodies |
| Parallel joints | Soft alignment, torque limiting, static/dynamic bodies |
| Supported contact/joint overflow | Serial GPU execution in deterministic upstream order |
| Mixed distance/parallel colors and overflow | Supported with type-dense buffers and ordered descriptors |

## Explicit CPU boundary

The following remain CPU work:

- broad phase, narrow phase, and manifold generation;
- contact and joint preparation;
- broad-phase tree mutation and pair generation, events, islands, sleeping, and CCD;
- recording, queries, topology mutation, and public API calls;
- filter, motor, prismatic, revolute, spherical, weld, and wheel joint solving;
- any joint requesting force/torque threshold events.

An unsupported joint causes the constrained solve to remain on the CPU for that
step. If the body threshold is met, the standalone position stage may still run
on Metal. Telemetry distinguishes these routes.

## Geometry and contact coverage

GPU kernels consume prepared convex-wide and scalar mesh constraints rather
than shape geometry. Consequently the supported contact solve can cover
contacts originating from Box3D's convex and mesh/height-field collision paths,
provided preparation selected a supported representation and no unsupported
overflow form is present. Collision detection itself remains CPU-side.

Body and awake-shape finalization have an experimental, separately opt-in Metal
path for rotation, origin offset, motion/sleep metrics, world-space inverse
inertia, speculative bounds, and fat-AABB enlargement. The CPU still applies
the flat shape results and performs pointer-rich dynamic-tree work. This path is
not enabled by `b3World_EnableMetal` alone because current whole-world
measurements are slower.

## Double precision

Double-precision world positions remain supported. Shape AABBs explicitly fall
back to the CPU so far-world translation retains Box3D's outward-rounded
binary64-add-then-float-narrow contract. `b3BodyState` and Metal constraint
arithmetic are float, matching the baseline solver-state precision. A dedicated
double-precision build runs the Metal differential suite.

## Failure behavior

Metal initialization failure returns `false` from `b3World_EnableMetal`; the
world remains usable. Unsupported constraints and dispatch failures return to
the CPU path and increment the appropriate fallback counter. Disabling Metal
releases its resources without destroying the world.
