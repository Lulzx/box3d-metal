# M4 Pro indexed cold-contact topology — 2026-09-03

This checkpoint replaces the compact 16-byte cold first-touch transition stream
with an 8-byte contact-ID-indexed table. Metal clears the table with a blit in
the same command buffer and scatters `{generation, pointCount}` directly into
the canonical contact slot. A complete event-free cold batch is validated
before mutation and committed once in ascending contact-ID order.

The direct commit retains Box3D's existing island-link, greedy graph-color, and
awake-set mutation semantics. It removes every per-worker contact-state bitset
clear, the worker-bitset union, and the second serial set-bit traversal. Mixed
or unsupported batches retain the original CPU exception and fallback path.
Recycled creation input order `2,1,0` still produces graph order `0,1,2`.

## Whole-world cold result

Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB, eight Box3D workers, Release,
single precision, spheres, four substeps, one cold step per independently
created world. Three sequential trials were taken at each size; medians are
wall-clock `b3World_Step` times.

| Contacts | Previous GPU median | Indexed/direct median | Change | Shared topology bytes | Direct topology CPU median |
|---:|---:|---:|---:|---:|---:|
| 131,072 | 86.685 ms | 84.335 ms | -2.7% | 1,048,576 | 7.995 ms |
| 262,144 | 164.411 ms | 158.536 ms | -3.6% | 2,097,152 | 15.504 ms |

Each measured row reports zero CPU collision contacts, zero manifold-exception
bytes, zero contact-state-bitset bytes, and a direct-commit count equal to the
contact count. The shared topology boundary is half its previous size. The
explicit CPU timing also shows the next bottleneck: island and graph mutation
still scale linearly and should move behind a device-resident topology epoch.

## Validation

- Float and double/VF64 warning-as-error Metal suites pass.
- Exact CPU/GPU pair, graph-color, island, and recycled-ID ordering checks pass.
- The double build uses the vendored exact IEEE-754 Metal implementation from
  VF64-metal commit `729021777455da72db8809d9ef1269c677d88b3f`.
- The complete CPU Release suite and full Metal AddressSanitizer suite pass.
- Float and double/VF64 Metal UndefinedBehaviorSanitizer suites pass with
  recovery disabled.

No PDF artifact is generated or modified by this checkpoint.
