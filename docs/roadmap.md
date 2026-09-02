# Roadmap and limitations

## Current limitations

- Raw dynamic-tree traversal, leaf update, and internal refit have an
  experimental Metal path, but CPU topology, candidate filtering, and narrow
  phase remain.
- Manifold and constraint preparation remain CPU-side.
- Body and awake-shape finalization have an experimental Metal path, but the CPU
  still owns topology mutation, sleeping/island
  mutation, events, and CCD. Successful resident refits now use a stable
  enlarged-only GPU stream and bypass the full-result rescan, CPU enlarged-body
  reduction, and body/shape-list traversal during proxy bookkeeping. Prior
  Metal fat bounds are authoritative across revision-stable steps. The full
  64-byte result is private; a bounded collision-free, non-CCD route avoids its
  blit/apply and selectively synchronizes public queries or route changes.
- Shape geometry, ids, filters, and proxy keys are still packed by walking awake
  body shape lists every step.
- Only distance and parallel joints stay in the GPU-resident constraint graph.
- Constraint joint records are packed/unpacked each step.
- Body state crosses the CPU/GPU ownership boundary once per world step.
- Small workloads are dominated by fixed submission and synchronization cost.
- Metal mode is tolerance-equivalent rather than cross-platform bit deterministic.

## Evidence-led next stages

1. Replace per-step `b3MetalPackShapeInputs` body/shape-list traversal with a
   persistent shape registry updated by topology and geometry mutators.
2. Retain body and supported joint state across world steps, reading back only
   public/event slices needed by the CPU.
3. Add remaining high-value joint types one at a time with mode matrices,
   overflow tests, static-body tests, and whole-world benchmarks.
4. Move shape-specialized narrow phase incrementally, keeping complex or rare
   geometry on an explicit CPU fallback path.
5. Add per-Apple-GPU-family benchmark records before considering automatic
   thresholds.

## Non-goals without new evidence

- Claiming universal speedups.
- Hiding unsupported paths behind relabeled telemetry.
- Replacing deterministic serial overflow with unsafe parallel writes.
- Treating unified memory as free synchronization.
- Promising compatibility with arbitrary future Box3D revisions.
