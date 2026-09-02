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
preserving upstream candidate order before the unchanged CPU filtering/contact
callback. Unchanged tree snapshots remain in persistent Metal storage, while
any CPU bounds or topology mutation invalidates them. Both stages remain off by
default while the CPU still owns tree mutation and consumes shared result
streams.

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
new whole-world timing is published from the loaded development machine.

Small workloads remain CPU-favorable. Metal is explicitly enabled per world
with a caller-selected body threshold.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
