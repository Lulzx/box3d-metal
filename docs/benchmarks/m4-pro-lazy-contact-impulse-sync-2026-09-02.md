# Lazy contact-impulse synchronization (2026-09-02)

## Scope

Successful resident convex solves no longer run Box3D's all-contact CPU
`b3StoreImpulses_Convex` traversal. The 80-byte contact-ID result table remains
GPU-authored and authoritative. Contact, body, and shape data APIs materialize
only requested manifolds. Contact-force debug drawing and world snapshots are
explicit synchronization boundaries because they consume public impulse state.

Hit events use a compact exception path. The existing CPU narrow-phase input
pack records IDs only for supported contacts whose shapes requested hit events;
there is no second graph traversal. After the GPU solve, one store block visits
that list, validates contact and result generations, synchronizes matching
features, and sets the existing per-worker event bits. Final event construction
therefore keeps upstream contact-ID order. Worlds with no hit-event requests
touch no contacts in the store stage.

Before each new solver route is selected, authority from the previous GPU result
is invalidated after collision has consumed it for warm starts. A CPU fallback
therefore cannot be overwritten later by stale resident output.

## Evidence

The 81-contact sphere/capsule differential performs four resident preparation
dispatches, one schedule pack and three schedule reuses, four CPU store
bypasses, and zero event or public-manifold synchronizations. CPU/GPU error
remains `5.96e-08` for transforms and `3.58e-07` for velocity.

The resident impact fixture produces the same hit event as the CPU oracle while
reporting exactly one store bypass, one event-exception synchronization, and one
total contact synchronization. The warm-start fixture deliberately poisons the
CPU mirror and verifies contact, body, and shape data APIs independently restore
the GPU-authored point values; recording snapshot capture independently restores
the same poisoned mirror. Its two simulation steps report two store bypasses,
zero event syncs, four explicit public/snapshot syncs, and `4.47e-08` velocity
error. The unsupported-joint fallback exposes no resident result afterward.

These are correctness and ownership counters. No loaded-host wall-clock result
is promoted to performance evidence; the updated resident-contact harness emits
store-bypass and synchronization counts for the next quiet-host comparison.
A loaded-host 512-contact smoke with eight warmups and two measured steps
reported ten preparation dispatches, one schedule pack, nine schedule reuses,
ten store bypasses, and zero event/public synchronizations. Its timing is not
published as performance evidence.
