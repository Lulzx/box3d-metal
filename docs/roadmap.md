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
  an 80-byte post-solve record per active contact. Successful resident steps
  bypass the all-contact CPU store; compact hit-enabled exceptions and explicit
  public/debug/snapshot consumers synchronize individual records. The
  contact-ID lane schedule remains resident across unchanged graph revisions
  and exact counts. Stable default-callback contacts also refresh their
  152-byte preparation records directly in the manifold scatter after the CPU
  seed. CPU manifold application, graph construction, callback/recycling
  exceptions, final event ordering, and topology remain.
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

Resident contact results now carry warm starts by contact generation and feature
ID. Public-manifold synchronization is lazy, and hit events use a compact
exception list assembled during narrow-phase packing.

1. Retain body and supported joint state across world steps, reading back only
   public/event slices needed by the CPU.
2. Add remaining high-value joint types one at a time with mode matrices,
   overflow tests, static-body tests, and whole-world benchmarks.
3. Move deterministic contact/topology mutation on-device now that ordinary
   first-touch geometry and CPU-manifold allocation are removed, while keeping
   the CPU oracle and ordered fallback boundary.
4. Add per-Apple-GPU-family benchmark records before considering automatic
   thresholds.

## Non-goals without new evidence

- Claiming universal speedups.
- Hiding unsupported paths behind relabeled telemetry.
- Replacing deterministic serial overflow with unsafe parallel writes.
- Treating unified memory as free synchronization.
- Promising compatibility with arbitrary future Box3D revisions.
