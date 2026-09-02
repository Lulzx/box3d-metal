# Box3D Metal

An opt-in Apple Silicon Metal compute backend for
[Box3D](https://github.com/erincatto/box3d), distributed as a reproducible
overlay rather than a fork containing the entire upstream source tree.

The backend keeps body state GPU-resident across supported solver phases and
accelerates fused integration, colored convex and mesh contacts, distance
joints, parallel joints, restitution, and deterministic graph overflow. The
original CPU implementation remains the behavioral reference and fallback.
An experimental, separately opt-in finalization kernel also computes final
rotation, origin offset, sleep-motion metrics, world-space inverse inertia, and
awake-shape AABBs. A second experimental opt-in traverses Box3D's existing
dynamic trees and performs deterministic candidate compaction on Metal,
preserving upstream candidate order. Resident moved-proxy marks and shape
metadata reject self/duplicate, same-body, sensor, and built-in filter pairs;
a revisioned mirror of Box3D's pair set suppresses existing non-compound
contacts. Compounds, joints, custom filters, and contact creation retain the CPU
path. Unchanged tree snapshots remain in persistent Metal storage, while
supported shape motion now updates leaves and refits internal bounds on-device.
Enlarged proxy bookkeeping consumes a stable GPU-compacted subset rather than
rescanning every result or walking body shape lists again.
Topology changes and unsupported CPU mutations invalidate the snapshot.
Double-precision worlds use VF64 exact software binary64 translation and
directed float narrowing for conservative far-world AABBs. On unchanged
resident steps, the prior Metal fat bounds—not the CPU mirror—are the next
dispatch's containment input; tree revisions fail closed to a CPU reseed.
The full 64-byte shape result now stays in private Metal storage. A bounded
collision-free, non-CCD route avoids its full blit and CPU apply; public AABB
queries stage individual records, while route changes synchronize explicitly.
Revision-stable steps also reuse persistent 72-byte shape-input records after
an exact awake-body id check, skipping shape counting, body shape-list traversal,
local AABB computation, and record writes. Both stages remain off by default;
CPU topology and cold/topology registry rebuilds are still in the path.
The shape-specialized narrow-phase route now batches sphere-sphere,
capsule-sphere, capsule-capsule, and bounded compact hull-sphere local manifold
geometry in one Metal command buffer, including ordered two-point capsule
manifolds and feature ids. Sphere/capsule endpoints and radii plus compact hull
descriptors live in persistent Metal buffers; identical hull point, plane, and
boundary-triangle streams are content-deduplicated. A body-id registry retains
static and awake rotations plus VF64-capable world translations for collision.
Revision-stable dispatches reuse both registries. Each 16-byte contact input
contains eligibility and two shape ids. Full 80-byte outputs stay in private
Metal storage; a deterministic scan/prefix/scatter pass returns only active
results, tagged by original contact index, in the same command buffer. The
scatter also rotates normals and points into world axes, so CPU workers
lower-bound once per range while retaining manifold persistence,
material and pre-solve callbacks, events, and graph/island state. High-aspect
and speculative hull-sphere contacts explicitly retain CPU GJK; other shape
pairs remain on the CPU. Double worlds use the vendored VF64 exact subtraction
before narrowing relative translations to float.
The same scatter writes active finalized records into a private table indexed
by Box3D contact id. This adds no steady-path readback or dispatch; explicit
table staging exists only for validation and fallback diagnostics.
Transient per-contact ownership now carries that authority through persistence,
and topology into solver setup. Pre-solve callbacks remain CPU-owned. When every
colored convex contact is resident-authoritative and no convex overflow exists,
a Metal kernel now prepares Erin's SIMD-wide contact constraints in the existing
solver command buffer. CPU persistence/material work writes a generation-tagged
144-byte record once into a contact-ID table during the existing collision pass.
Solver submission bulk-copies only a deterministic four-byte ID schedule per
SIMD lane and no longer dereferences contacts to repack those records. Mixed,
recycled, callback, overflow, and unsupported routes fail closed, including
explicit CPU prepare-on-fallback recovery.
After restitution, Metal extracts one 80-byte impulse record per active contact
into a generation-tagged contact-ID table. CPU public-manifold and hit-event
synchronization keeps upstream order while consuming that compact table instead
of the 1,696-byte SIMD-wide records. Invalid and unsupported routes retain the
original CPU store path.

## Quick start

Requirements: Apple Silicon, macOS, Xcode command-line tools, CMake, Ninja, and
Git.

```sh
git clone https://github.com/Lulzx/box3d-metal.git
cd box3d-metal
./scripts/bootstrap.sh ../box3d-metal-worktree
../box3d-metal-worktree/build/metal-release/bin/test MetalTest
../box3d-metal-worktree/build/metal-release/bin/metal_demo
```

The bootstrap script clones the pinned upstream revision, verifies the patch,
applies it, configures a Release build, and builds tests, demos, and benchmarks.
Nothing from the upstream Box3D checkout is stored in this repository.

## Documentation

Start with the [documentation index](docs/index.md). Compact printable guides
are under [`docs/pdf`](docs/pdf/).

- [Quick start](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Compatibility contract](docs/compatibility.md)
- [Correctness and validation](docs/validation.md)
- [Performance evidence](docs/performance.md)
- [Distribution and reproducibility](docs/reproducibility.md)
- [Roadmap and limitations](docs/roadmap.md)

## Measured M4 Pro results

Whole-world or end-to-end measurements, including synchronization and relevant
CPU work:

| Workload | Largest demonstrated result |
| --- | ---: |
| Fused four-substep integration primitive | 10.584x at 524,288 bodies |
| Unconstrained whole world | 1.146x at 524,288 bodies |
| Scalar multi-manifold mesh contacts | 1.646x at 131,072 bodies |
| Convex contacts | 1.098x at 262,144 bodies |
| Distance joints | 1.158x at 524,288 bodies |
| Parallel joints | No stable whole-world crossover demonstrated |
| Experimental GPU finalization | Correct, but 27% slower at the 524,288-body paired median |
| GPU shape finalization | Correct, but 17.9% slower at 524,288 shapes |
| Experimental GPU tree traversal | 1.068x at 524,288 shapes; small worlds regress |

The tree-traversal speedup is historical evidence for the earlier CPU-prefix
implementation. The current on-device scan has exact-order validation, but no
new whole-world timing is published from the loaded development machine. The
resident-refit and VF64 checkpoint likewise publishes correctness evidence only.
The private-result/selective-sync checkpoint also publishes correctness evidence
only because the development host was loaded.
The persistent-input checkpoint likewise publishes route/correctness evidence
only because the host remained loaded.
The resident pair-filter checkpoint publishes exact-order correctness and
metadata-residency evidence only; the host was still loaded.
The existing-pair checkpoint likewise publishes lifecycle and residency
evidence only, not timing from the loaded host.
The bounded hull-sphere checkpoint likewise publishes CPU-oracle,
deterministic replay, mixed-fallback, two-point ordering, and VF64 far-world
evidence only. It does not publish loaded-host timing.
The resident hull-geometry checkpoint publishes exact upload/reuse/rebuild and
deduplication evidence only; it does not convert loaded-host kernel timings into
a whole-world performance claim.
The resident shape-geometry checkpoint adds primitive-mutation invalidation and
120-byte input evidence under the same no-loaded-host-timing boundary.
The resident body-transform checkpoint reduces that input to 16 bytes and adds
step/teleport/replay invalidation evidence, again without a loaded-host timing claim.
The private manifold-result checkpoint adds active-only deterministic readback
and range-linear CPU consumption under the same no-loaded-host-timing boundary.
The manifold-finalization checkpoint fuses world-axis orientation into that
scatter, again publishing correctness rather than loaded-host timing.
The resident manifold-table checkpoint establishes stable contact-id addressing
under the same correctness-only timing boundary.
The solver-ownership checkpoint establishes the fail-closed preparation gate;
the resident contact-preparation checkpoint now uses it to skip CPU preparation
arithmetic for complete supported sets. The metadata-residency checkpoint also
removes the dedicated solver-time contact traversal and 144-byte lane stream;
CPU persistence/table writes and graph scheduling remain. Compact post-solve
extraction reduces the 81-contact CPU impulse-input surface from 35,616 to 6,480
bytes while retaining public manifolds and matching hit events. It adds no
loaded-host speedup claim.

Small workloads remain CPU-favorable. Metal is explicitly enabled per world
with a caller-selected body threshold.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
