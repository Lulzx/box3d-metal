# Roadmap and limitations

## Current limitations

- Dynamic-tree traversal, leaf update, internal refit, moved-proxy de-duplication,
  built-in same-body/sensor/shape filtering, and existing non-compound pair
  suppression have an experimental Metal path,
  but CPU topology, joint/custom/compound filtering, and contact creation remain.
- Sphere-sphere, capsule-sphere, capsule-capsule, and bounded compact
  hull-sphere local manifold geometry has a batched Metal path. Speculative and
  high-aspect hull-sphere, other hull pairs, meshes, height fields, compounds,
  manifold state application, and constraint preparation remain CPU-side.
- Supported sphere/capsule records and content-deduplicated compact hull geometry
  are retained across revision-stable dispatches. Static/awake/sleeping body
  transforms are retained by body id with VF64 positions. Full manifold results
  are private and only active, world-oriented records return through a shared
  ordered stream. Active finalized records also live in a private table indexed
  by contact id. A transient ownership marker now reaches solver setup and
  exposes complete SIMD-wide coverage only when no CPU/recycling exception is
  present. Complete colored resident convex sets are prepared on Metal;
  pre-solve callbacks, mixed/recycled sets, and convex overflow remain CPU-side.
  Post-persistence records are retained in a generation-tagged contact-ID table,
  reducing solver submission to a four-byte lane schedule. Metal also extracts
  an 80-byte post-solve record per active contact; CPU public-manifold and event
  synchronization consumes those contact-ID records instead of wide solver
  constraints. The contact-ID lane schedule remains resident across unchanged
  graph revisions and exact counts. CPU persistence, graph construction,
  events, and topology remain.
- Body and awake-shape finalization have an experimental Metal path, but the CPU
  still owns topology mutation, sleeping/island
  mutation, events, and CCD. Successful resident refits now use a stable
  enlarged-only GPU stream and bypass the full-result rescan, CPU enlarged-body
  reduction, and body/shape-list traversal during proxy bookkeeping. Prior
  Metal fat bounds are authoritative across revision-stable steps. The full
  64-byte result is private; a bounded collision-free, non-CCD route avoids its
  blit/apply and selectively synchronizes public queries or route changes.
- Revision-stable steps reuse persistent shape geometry, ids, filters, proxy
  keys, and local bounds after an exact awake-body order check. Cold starts and
  invalidated registries still walk awake body shape lists and repack.
- Only distance and parallel joints stay in the GPU-resident constraint graph.
- Constraint joint records are packed/unpacked each step.
- Body state crosses the CPU/GPU ownership boundary once per world step.
- Small workloads are dominated by fixed submission and synchronization cost.
- Metal mode is tolerance-equivalent rather than cross-platform bit deterministic.

## Evidence-led next stages

1. Make public-manifold/event synchronization lazy or exception-only. Move
   supported persistence fully on-device while returning only compact callback,
   event, topology, and unsupported-geometry exceptions.
2. Retain body and supported joint state across world steps, reading back only
   public/event slices needed by the CPU.
3. Add remaining high-value joint types one at a time with mode matrices,
   overflow tests, static-body tests, and whole-world benchmarks.
4. Move deterministic contact creation only after the geometry and manifold
   residency boundaries are proven, keeping topology mutation ordered.
5. Add per-Apple-GPU-family benchmark records before considering automatic
   thresholds.

## Non-goals without new evidence

- Claiming universal speedups.
- Hiding unsupported paths behind relabeled telemetry.
- Replacing deterministic serial overflow with unsafe parallel writes.
- Treating unified memory as free synchronization.
- Promising compatibility with arbitrary future Box3D revisions.
