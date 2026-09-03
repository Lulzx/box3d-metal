# Documentation index

This index separates what is implemented, what is measured, what falls back,
and how to reproduce it. Read the short path for adoption or the evidence path
for review.

## Adoption path

1. [Quick start](quickstart.md) - reconstruct, build, enable, and inspect Metal.
2. [Compatibility contract](compatibility.md) - decide whether your world stays
   on the GPU-resident constraint path.
3. [Performance](performance.md) - choose a threshold from measured evidence,
   then benchmark your workload.
4. [Troubleshooting](troubleshooting.md) - diagnose initialization, fallback,
   build, and performance issues.

## Technical path

1. [Architecture](architecture.md) - command graph, shared-memory ownership,
   kernels, graph colors, and overflow handling.
2. [Correctness and validation](validation.md) - CPU reference, differential
   gates, sanitizers, build matrix, and numerical limits.
3. [Reproducibility and distribution](reproducibility.md) - pinned provenance,
   patch construction, clean-clone verification, and repository contents.
4. [Roadmap and limitations](roadmap.md) - remaining CPU stages and evidence-led
   next steps.

## Printable guides

- [Quick Start PDF](pdf/box3d-metal-quickstart.pdf)
- [Architecture PDF](pdf/box3d-metal-architecture.pdf)
- [Compatibility and Validation PDF](pdf/box3d-metal-compatibility-validation.pdf)
- [M4 Pro Performance PDF](pdf/box3d-metal-m4-pro-performance.pdf)

## Benchmark records

- [Position integration](benchmarks/m4-pro-position-integration-2026-09-01.md)
- [Fused integration](benchmarks/m4-pro-fused-integration-2026-09-01.md)
- [Convex contacts](benchmarks/m4-pro-convex-contacts-2026-09-01.md)
- [Mesh contacts](benchmarks/m4-pro-mesh-contacts-2026-09-01.md)
- [Distance joints](benchmarks/m4-pro-distance-joints-2026-09-01.md)
- [Parallel joints](benchmarks/m4-pro-parallel-joints-2026-09-02.md)
- [Experimental body finalization](benchmarks/m4-pro-finalization-2026-09-02.md)
- [Awake-shape AABB finalization](benchmarks/m4-pro-shape-finalization-2026-09-02.md)
- [GPU pair generation](benchmarks/m4-pro-pair-generation-2026-09-02.md)
- [On-device pair compaction](benchmarks/m4-pro-pair-prefix-2026-09-02.md)
- [Resident pair filtering](benchmarks/m4-pro-resident-pair-filtering-2026-09-02.md)
- [Resident existing-pair suppression](benchmarks/m4-pro-resident-existing-pairs-2026-09-02.md)
- [Enlarged-shape compaction](benchmarks/m4-pro-shape-compaction-2026-09-02.md)
- [Private shape results](benchmarks/m4-pro-private-shape-results-2026-09-02.md)
- [Persistent shape-input registry](benchmarks/m4-pro-shape-input-registry-2026-09-02.md)
- [Persistent hull-geometry registry](benchmarks/m4-pro-resident-hull-geometry-2026-09-02.md)
- [Persistent shape-geometry registry](benchmarks/m4-pro-resident-shape-geometry-2026-09-02.md)
- [Persistent body-transform registry](benchmarks/m4-pro-resident-body-transforms-2026-09-02.md)
- [Private manifold results](benchmarks/m4-pro-private-manifold-results-2026-09-02.md)
- [Manifold finalization](benchmarks/m4-pro-manifold-finalization-2026-09-02.md)
- [Resident manifold table](benchmarks/m4-pro-resident-manifold-table-2026-09-02.md)
- [Resident solver ownership](benchmarks/m4-pro-resident-solver-ownership-2026-09-02.md)
- [Resident contact preparation](benchmarks/m4-pro-resident-contact-preparation-2026-09-02.md)
- [Contact-prepare metadata residency](benchmarks/m4-pro-contact-prepare-residency-2026-09-02.md)
- [Contact-impulse result residency](benchmarks/m4-pro-contact-impulse-residency-2026-09-02.md)
- [Resident contact schedule](benchmarks/m4-pro-resident-contact-schedule-2026-09-02.md)
- [Resident warm-start carry](benchmarks/m4-pro-resident-warm-start-2026-09-02.md)
- [Lazy contact-impulse synchronization](benchmarks/m4-pro-lazy-contact-impulse-sync-2026-09-02.md)
- [Resident contact finalization](benchmarks/m4-pro-contact-finalization-residency-2026-09-02.md)
- [Device contact-prepare refresh](benchmarks/m4-pro-device-contact-prepare-refresh-2026-09-02.md)
- [Resident collision-application bypass](benchmarks/m4-pro-resident-collision-bypass-2026-09-02.md)
- [Contact exception compaction](benchmarks/m4-pro-contact-exception-compaction-2026-09-02.md)
- [Resident contact input/order registry](benchmarks/m4-pro-contact-input-residency-2026-09-02.md)
- [Contact-state traversal bypass](benchmarks/m4-pro-contact-state-traversal-bypass-2026-09-02.md)
- [Hit-event bitset residency](benchmarks/m4-pro-hit-event-bitset-residency-2026-09-02.md)
- [Private cold first-touch transitions](benchmarks/m4-pro-private-first-touch-2026-09-03.md)
- [Deferred CPU manifolds](benchmarks/m4-pro-deferred-cpu-manifolds-2026-09-03.md)

## Repository map

| Path | Purpose |
| --- | --- |
| `manifest.env` | Canonical upstream URL, pinned revision, patch path |
| `patches/box3d-metal.patch` | Complete source overlay |
| `scripts/bootstrap.sh` | Clone, verify, apply, configure, and build |
| `scripts/verify-clean.sh` | End-to-end reconstruction and validation |
| `scripts/run-benchmarks.sh` | Rebuild and run the benchmark executables |
| `tools/build_pdfs.py` | Deterministic compact PDF generator |
| `docs/pdf/` | Render-verified printable guides |
