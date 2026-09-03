# Private first-touch contact transitions on M4 Pro

## Scope

This checkpoint removes the full shared manifold stream and CPU collision-worker
pass from the completely cold, ordinary convex-contact route. Metal keeps the
240-byte finalized manifold in its private contact-ID table and emits a
deterministically compacted 16-byte topology record:

```text
{ contactId, contactGeneration, inputIndex, pointCount }
```

The CPU validates the entire stream before mutation, allocates a zeroed
one-manifold structural placeholder, patches the already GPU-authored prepare
record with that pointer, and feeds the existing contact-state bitset. Box3D's
ascending contact-ID island and graph transition order therefore remains the
topology oracle. Public queries, debug/snapshot access, Metal disable, graph
overflow, and CPU solver fallback materialize the private manifold on demand.

The private path is deliberately limited to a complete cold bootstrap with
default materials, zero recycle distance, awake/static bodies, no recording,
and no contact, hit, or pre-solve events. Unsupported, separated, fast, or
otherwise unsafe contacts retain the 240-byte CPU exception path.

## Device and build

- Apple M4 Pro
- macOS 26.7 (25G227)
- Apple clang 21.0.0 (clang-2100.1.1.101)
- arm64 Release, Metal enabled, eight Box3D workers, four substeps
- whole `b3World_Step` wall time, one cold step per fresh world
- baseline: clean detached `56e5dea`
- candidate: working tree based on `56e5dea`

Command shape:

```sh
BOX3D_METAL_RESIDENT_CONTACT_COLD_PAIR=1 \
BOX3D_METAL_RESIDENT_CONTACT_COUNT=131072 \
build/metal-werror/bin/metal_resident_contact_benchmark
```

## Whole-world cold result

GPU samples are independent fresh-process runs. Both medians use three runs.

| Contacts | Baseline GPU samples (ms) | Private-first-touch samples (ms) | Median change |
|---:|---|---|---:|
| 131,072 | 115.435, 109.273, 115.063 | 95.659, 93.112, 90.757 | **-19.1%** |
| 262,144 | 233.540, 215.461, 217.687 | 173.885, 176.261, 168.888 | **-20.1%** |

The route is still slower than the CPU oracle in this cold one-step workload.
Median per-process CPU/GPU ratios improve from 0.668x to 0.874x at 131,072 and
from 0.742x to 0.904x at 262,144. This is progress on the measured
regression, not a crossover claim.

## Ownership evidence

| Contacts | Old shared manifold bytes | New transition bytes | Full exception bytes | CPU collision contacts |
|---:|---:|---:|---:|---:|
| 131,072 | 31,457,280 | 2,097,152 | 0 | 0 |
| 262,144 | 62,914,560 | 4,194,304 | 0 | 0 |

The private manifold and prepare tables remain 240 and 224 bytes per contact,
respectively; this change removes their cold geometry from the CPU-visible
result boundary. The CPU still allocates one zeroed structural manifold per
touching contact because the current constraint-graph and public API invariants
require a non-null manifold pointer. Removing that allocation is a later,
broader invariant change.

## Correctness gates

- Single-precision and double/VF64 Release Metal suites pass.
- Float and double/VF64 UBSan Metal suites pass with halt-on-error.
- ASan Metal suite passes with halt-on-error; leak detection is unavailable on
  this macOS ASan runtime.
- The complete CPU Release test suite passes.
- Differential tests prove exact touching flags, graph colors, color body
  bitsets, graph contact order, body simulation indices, and lazy manifold
  materialization.
- Recycled LIFO creation order `2,1,0` still produces ascending serial graph
  order `0,1,2`.
- Corrupt bootstrap identity and revision mismatch expose no transition
  authority and recover through the CPU pack.
- Hit-event contacts retain one full 240-byte exception.
