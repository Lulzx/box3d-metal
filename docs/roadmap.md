# Roadmap and limitations

## Current limitations

- Broad phase and narrow phase remain CPU-side.
- Manifold and constraint preparation remain CPU-side.
- Body-finalization arithmetic has an experimental Metal path, but bounds,
  sleeping/island mutation, events, and CCD remain CPU-side.
- Only distance and parallel joints stay in the GPU-resident constraint graph.
- Constraint joint records are packed/unpacked each step.
- Body state crosses the CPU/GPU ownership boundary once per world step.
- Small workloads are dominated by fixed submission and synchronization cost.
- Metal mode is tolerance-equivalent rather than cross-platform bit deterministic.

## Evidence-led next stages

1. Move shape finalization and AABB generation into the existing command graph,
   eliminating the shared arithmetic-result stream and CPU shape traversal.
2. Implement GPU pair generation and broad phase after bounds ownership is
   resolved, using deterministic radix-sort/compaction designs suitable for
   Apple GPUs.
3. Retain body and supported joint state across world steps, reading back only
   public/event slices needed by the CPU.
4. Add remaining high-value joint types one at a time with mode matrices,
   overflow tests, static-body tests, and whole-world benchmarks.
5. Move shape-specialized narrow phase incrementally, keeping complex or rare
   geometry on an explicit CPU fallback path.
6. Add per-Apple-GPU-family benchmark records before considering automatic
   thresholds.

## Non-goals without new evidence

- Claiming universal speedups.
- Hiding unsupported paths behind relabeled telemetry.
- Replacing deterministic serial overflow with unsafe parallel writes.
- Treating unified memory as free synchronization.
- Promising compatibility with arbitrary future Box3D revisions.
