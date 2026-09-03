# Deferred CPU manifolds on M4 Pro

## Scope

This checkpoint removes the remaining per-contact CPU `b3Manifold` allocation
from a completely cold, ordinary resident-contact step. Metal keeps the
240-byte finalized manifold and 224-byte prepare record in contact-ID-indexed
private tables. The CPU receives only the validated 16-byte touching transition
and installs the logical topology state:

```text
manifoldCount = 1
manifolds = NULL
flags = resident | stale
```

That representation is restricted to awake, supported convex contacts whose
device identity and generation are current. Graph coloring, island linking, and
the Metal solver consume contact IDs and the logical count, so they do not need
a CPU geometry allocation. Public contact/body/shape queries, force debug draw,
recording snapshots, sleep, Metal disable, overflow, and CPU solver fallback
materialize a checked CPU mirror on demand. Separation and destruction can
discard an unmaterialized logical contact directly.

## Device and build

- Apple M4 Pro
- macOS 26.7 (25G227)
- Apple clang 21.0.0 (clang-2100.1.1.101)
- arm64 Release, Metal enabled, eight Box3D workers, four substeps
- whole `b3World_Step` wall time, one cold step per fresh process/world
- prior checkpoint: source commit `0715079`
- candidate: source commit `2fe7d51`

Command shape:

```sh
BOX3D_METAL_RESIDENT_CONTACT_COLD_PAIR=1 \
BOX3D_METAL_RESIDENT_CONTACT_COUNT=131072 \
build/metal-release/bin/metal_resident_contact_benchmark
```

## Whole-world cold result

Each sample is an independent fresh-process run. Medians use three runs.

| Contacts | Allocating-placeholder samples (ms) | Deferred-manifold samples (ms) | Median change |
|---:|---|---|---:|
| 131,072 | 95.659, 93.112, 90.757 | 79.277, 86.685, 89.614 | **-6.9%** |
| 262,144 | 173.885, 176.261, 168.888 | 170.563, 164.411, 158.582 | **-5.4%** |

The CPU-oracle medians in the new runs are 77.789 ms and 159.283 ms. The Metal
route therefore remains about 11.4% slower at 131,072 and 3.2% slower at
262,144. One 262,144 sample crossed over, but three cold samples do not support
a stable crossover claim. The evidence is a smaller first-touch regression.

## Ownership evidence

| Contacts | Touching transitions | Transition bytes | CPU collision contacts | Manifold syncs | Resident contacts without CPU manifolds |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 131,072 | 2,097,152 | 0 | 0 | 131,072 |
| 262,144 | 262,144 | 4,194,304 | 0 | 0 | 262,144 |

The private manifold and prepare tables remain device-resident. The result
stream is unchanged from the previous checkpoint; the measured gain comes from
removing the CPU allocator work and zero-fill for ordinary first touches.

## Correctness gates

- Single-precision and double/VF64 strict Metal suites pass.
- The cold topology differential retains exact contact IDs, generations,
  touching flags, graph colors, body bitsets, graph order, islands, and body-sim
  indices while every unobserved GPU contact has a null CPU pointer.
- Four resident solve steps complete with 81 logical manifolds and zero CPU
  geometry allocations or synchronizations.
- One public query materializes exactly one requested contact and repeated
  access is idempotent.
- Unsupported-joint solver fallback bulk-materializes before CPU preparation.
- Sleep and Metal disable materialize before moving authority off the resident
  tables.
- Separation and live body destruction discard null-backed contacts safely.
- Recording snapshot creation materializes all resident logical manifolds
  before serialization, and invalid impulse authority rejects recording startup.
- The complete CPU Release suite passes.
- Float and double/VF64 UBSan Metal suites pass with halt-on-error.
- The ASan Metal suite passes with halt-on-error; leak detection is disabled on
  this macOS ASan run.
