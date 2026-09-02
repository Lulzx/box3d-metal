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
| Resident convex contact preparation | Complete colored sets prepared from the private contact-id table; CPU recovery on later solver fallback |
| Experimental broad phase | Resident leaf update/refit plus built-in filtered candidates in exact CPU visitation order |
| Experimental narrow phase | Sphere-sphere, capsule-sphere, capsule-capsule, and bounded compact hull-sphere local geometry; primitive records and deduplicated compact hull streams are retained across revision-stable dispatches |

## Explicit CPU boundary

The following remain CPU work:

- broad-phase topology mutation, joint/custom/compound filtering, and contact
  creation;
- unsupported narrow-phase pairs, speculative/high-aspect hull-sphere GJK, and
  manifold state application; per-contact eligibility/id validation and ordered
  manifold-result consumption;
- mesh, joint, callback, mixed/recycled convex, and convex-overflow preparation;
- events, islands, sleeping, and CCD;
- recording, queries, topology mutation, and public API calls;
- filter, motor, prismatic, revolute, spherical, weld, and wheel joint solving;
- any joint requesting force/torque threshold events.

An unsupported joint causes the constrained solve to remain on the CPU for that
step. If the body threshold is met, the standalone position stage may still run
on Metal. Telemetry distinguishes these routes.

## Geometry and contact coverage

The convex solver consumes constraints prepared on Metal when every colored
contact owns a current private-table result and no callback or convex overflow
exception exists. Normal and identity remain device-private. CPU persistence
writes generation-tagged metadata by contact ID; solver submission carries only
a four-byte ID schedule per SIMD lane. Other convex constraints and all scalar
mesh constraints arrive CPU-prepared. The supported contact solve
can therefore cover contacts originating from both accelerated convex pairs and
Box3D's CPU convex/mesh/height-field collision paths. Collision detection is
partially accelerated only for the experimental narrow-phase pairs listed above.

Body and awake-shape finalization have an experimental, separately opt-in Metal
path for rotation, origin offset, motion/sleep metrics, world-space inverse
inertia, speculative bounds, and fat-AABB enlargement. The CPU still applies
the flat shape results and performs pointer-rich shape bookkeeping. Resident
bounds feed the Metal tree refit directly. This path is
not enabled by `b3World_EnableMetal` alone because current whole-world
measurements are slower.

Dynamic-tree traversal has another separately opt-in Metal path. It queries
kinematic, static, and dynamic trees in upstream order and preserves each
tree's DFS leaf order. Metal rejects exact moved-proxy duplicates, same-body
pairs, sensors, and built-in group/category/mask failures from resident tables.
Metal also mirrors the pair set and suppresses existing non-compound contacts.
The CPU still owns compounds, joint overrides, custom callbacks,
deterministic contact-list construction, topology changes, and CPU fallback
rebuilds. Excessive tree depth or candidate volume and any dispatch or
allocation failure fall back to the complete CPU traversal for that step.

## Double precision

Double-precision world positions remain supported. VF64 exact integer binary64
helpers reproduce center-plus-delta-plus-origin translation and directed float
narrowing on Metal. A scale-aware local-float envelope keeps GPU AABBs
conservative against the CPU oracle at far-world coordinates. `b3BodyState`,
shape-local geometry, and Metal constraint arithmetic remain float, matching
the baseline solver-state precision. A dedicated double-precision build runs
the Metal differential suite.

## Failure behavior

Metal initialization failure returns `false` from `b3World_EnableMetal`; the
world remains usable. Unsupported constraints and dispatch failures return to
the CPU path and increment the appropriate fallback counter. Disabling Metal
releases its resources without destroying the world.
