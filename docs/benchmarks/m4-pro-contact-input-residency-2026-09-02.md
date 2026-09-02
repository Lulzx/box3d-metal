# M4 Pro contact input residency — 2026-09-02

## Scope

The 32-byte narrow-phase contact input and its deterministic graph/non-touching
order now remain in the Metal buffer across unchanged pair-set,
constraint-graph, and explicit eligibility revisions. Stable steps do not
allocate or gather a CPU contact-ID array and do not rewrite the input buffer.
Cold steps and contact, topology, callback, recording, recycling, hit-event, or
pre-solve eligibility changes rebuild from Erin Catto's CPU-owned structures.

Body-local solver indices and transient fast flags are deliberately not frozen
in the cached input. The per-step body-ID registry carries both. The manifold
scatter refreshes preparation indices from that registry, while the scan rejects
a resident bypass if either body is currently fast.

The exception scan also proves solver ownership when every convex graph contact
was stable and the graph revision did not change during collision state
processing. Solver setup then accumulates known color counts without
dereferencing each contact to recheck its resident flag. Mesh contacts, graph
topology, callbacks, events, allocation, and unsupported geometry remain CPU
boundaries.

## Differential evidence

The 81-contact sphere/capsule differential ran four steps with two input packs
and two reuses. The latest stable step wrote zero input bytes, emitted zero
manifold bytes, ran zero CPU collision contacts, and skipped 243 cumulative
per-contact solver coverage checks. Its transform and velocity errors remained
`2.69e-05` and `3.58e-07` respectively.

Hit-event mutation invalidated the registry once and produced one ordered
160-byte CPU exception. Adding one new touching contact invalidated it again and
produced one first-touch exception. Both retained the previous 80 or 81 stable
contacts on the device.

A dedicated mutation fixture reused the registry across an awake-body index
swap. The current body table supplied the new index, CPU/GPU velocity remained
within `3e-05`, and no exception was emitted. The same fixture then set the
current body fast without changing the cache key; the GPU emitted exactly one
CPU exception and did not count a resident bypass. This proves the cached input
does not hide transient CCD eligibility.

Float and double/VF64 warning-as-error Metal builds and focused Metal suites
passed. The complete float debug, AddressSanitizer, and
UndefinedBehaviorSanitizer suites passed, as did focused double/VF64 UBSan. The
non-Metal Release build and complete CPU test suite also passed. Those broader
gates caught and fixed both unguarded CPU-only exception-path references and a
null contact-order prefetch on a cached exception step. The VF64 implementation
remained pinned at commit
`729021777455da72db8809d9ef1269c677d88b3f`.

## Structural benchmark evidence

The Release resident-contact harness used eight warmups, five measured steps,
four substeps, and eight CPU workers:

| Contacts | Input packs | Input reuses | Latest input bytes | Collision bypasses | Coverage checks bypassed | Latest exceptions | Latest manifold bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 11 | 0 | 6,144 | 6,144 | 0 | 0 |
| 8,192 | 2 | 11 | 0 | 98,304 | 98,304 | 0 | 0 |

Each run processed contacts on the CPU only during its seed step. The second
pack follows the first-touch graph transition; every later stable dispatch
reuses the resident order and inputs.

Timing is not promoted. A Python process was consuming a full CPU core and the
host load averages were about 5 to 8. The observed wall-clock rows are useful
only as smoke tests, not as evidence for a new routing threshold.

## Next boundary

The stable supported convex path has removed the shared manifold stream, flat
collision task, contact-ID gather, input rewrite, solver ownership walk,
contact-preparation arithmetic, lane-schedule repack, and impulse-store walk.
The next performance gate is a quiet-host whole-world matrix. Its result should
decide whether command submission/synchronization or remaining CPU topology and
event work is the next dominant boundary before widening geometry support.
