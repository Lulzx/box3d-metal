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
