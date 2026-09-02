# Roadmap and limitations

## Current limitations

- Raw dynamic-tree traversal, leaf update, and internal refit have an
  experimental Metal path, but CPU topology, candidate filtering, and narrow
  phase remain.
- Manifold and constraint preparation remain CPU-side.
- Body and awake-shape finalization have an experimental Metal path, but the CPU
  still consumes flat shape results and owns tree mutation, sleeping/island
  mutation, events, and CCD.
- Only distance and parallel joints stay in the GPU-resident constraint graph.
- Constraint joint records are packed/unpacked each step.
- Body state crosses the CPU/GPU ownership boundary once per world step.
- Small workloads are dominated by fixed submission and synchronization cost.
- Metal mode is tolerance-equivalent rather than cross-platform bit deterministic.

## Evidence-led next stages

1. Make resident shape bounds authoritative for downstream GPU work and replace
   the flat per-shape CPU bookkeeping stream with selective synchronization.
   Leaf update, internal refit, stable prefixing, and compaction are on-device.
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
