# Private cold topology epoch on M4 Pro

Date: 2026-09-03

This checkpoint admits a cold, event-free, all dynamic-vs-static batch as a
device-private one-color contact schedule. The Metal solver consumes that
schedule directly without linking CPU islands or graph colors. CPU island and
graph topology is materialized lazily at the next observation, next-step,
mutation, or fallback boundary.

The route is deliberately narrow and fails closed. Joints, contact or hit or
pre-solve events, custom filter or friction or restitution callbacks,
recording, contact recycling, sleep, continuous collision, sensors,
unsupported topology (non dynamic-static pairs, repeated dynamic bodies,
unsupported geometry), and any pair, input, or graph revision mismatch reject
the private schedule and restore the canonical CPU topology before the
graph-based solver observes any constraints.

## Device and method

- Apple M4 Pro, 12 CPU cores and 16 GPU cores, 24 GB unified memory
- macOS 26.7 (25G227)
- Apple clang 21.0.0 (clang-2100.1.1.101)
- Release + Werror, eight Box3D workers, four substeps
- Whole-world cold sphere-contact step, wall-clock timing
- Baseline `efbdc6c`; candidate working tree built from that exact commit
- Seven alternating baseline/candidate runs at each size

The process-level samples are noisy, so both raw medians and the median of the
seven paired candidate-minus-baseline deltas are reported.

Reproduction:

```sh
cmake --build build/metal-werror --target metal_resident_contact_benchmark
BOX3D_METAL_RESIDENT_CONTACT_COLD_PAIR=1 BOX3D_METAL_RESIDENT_CONTACT_COUNT=131072 ./build/metal-werror/bin/metal_resident_contact_benchmark
BOX3D_METAL_RESIDENT_CONTACT_COLD_PAIR=1 BOX3D_METAL_RESIDENT_CONTACT_COUNT=262144 ./build/metal-werror/bin/metal_resident_contact_benchmark
```

The first-step `gpu_ms` column stays comparable with `efbdc6c`. After the
timed cold step the harness reads the profile, takes a second timed
`b3World_Step` so the deferred materialization runs, then reports the second
profile's `lastDeferredContactTopologyMaterializationMilliseconds` and the
second-step wall time.

## Cold device step

| Contacts | Baseline median | Candidate median | Raw median change | Paired median delta |
|---:|---:|---:|---:|---:|
| 131,072 | 83.317 ms | 65.805 ms | -21.0% | -18.644 ms |
| 262,144 | 153.860 ms | 119.452 ms | -22.4% | -32.280 ms |

The baseline step includes the CPU island/graph commit; the candidate step
does not. The candidate leaves `contact_topology_direct_commits` at zero and
`last_contact_transition_bytes` at zero on the cold step.

## Deferred materialization

| Contacts | Baseline eager topology CPU | Candidate deferred materialization | Candidate second step wall |
|---:|---:|---:|---:|
| 131,072 | 7.944 ms | 7.846 ms | 23.148 ms |
| 262,144 | 15.344 ms | 15.196 ms | 44.965 ms |

The baseline column is `last_contact_topology_cpu_ms` (the same work done
eagerly). The candidate column is the second-step
`lastDeferredContactTopologyMaterializationMilliseconds`, which commits the
prior epoch once in ascending contact-id order before the resident contact
path proceeds.

## Transport proof

| Contacts | `lastContactTransitionBytes` | Schedule private bytes | Summary shared bytes | `contact_topology_direct_commits` |
|---:|---:|---:|---:|---:|
| 131,072 | 0 | 524,288 | 32 | 0 |
| 262,144 | 0 | 1,048,576 | 32 | 0 |

Schedule bytes equal `4 * roundup(count, SIMD width)` with `B3_SIMD_WIDTH`
4. Summary bytes are the single 32-byte `b3MetalManifoldSummary` shared for
validation. The focused `MetalPrivateContactTopologyEpochTest` additionally
asserts transition bytes zero, immediate topology CPU zero, and exact CPU
topology after materialization.

## Correctness gates

- `build/metal-werror` full suite
- `build/metal-double-werror` full suite
- `build/metal-asan` full suite with `ASAN_OPTIONS=detect_leaks=0`
- `build/metal-ubsan` full suite with `UBSAN_OPTIONS=halt_on_error=1`
- `build/metal-double-ubsan` full suite with `UBSAN_OPTIONS=halt_on_error=1`
- `build/cpu-release` oracle suite
- `build/metal-debug` full suite including `MetalPrivateContactTopologyEpochTest`
- `build/metal-release` full suite

## Remaining bottleneck

The epoch covers only strict independent dynamic-static batches and
materializes CPU topology on the very next step. The next cut is extending
beyond strict dynamic-static independence and keeping topology resident
across steps instead of paying one materialization per cold epoch.
