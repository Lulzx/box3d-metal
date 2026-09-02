# M4 Pro contact exception compaction — 2026-09-02

## Scope

The Metal narrow phase now compacts only contacts that still require Erin
Catto's CPU collision path. A deterministic block scan, block prefix, and
scatter classify stable resident contacts separately from callback, topology,
first-touch, unsupported, and other fail-closed exceptions. All supported
contacts are finalized into the private contact-ID table; only ordered
exceptions enter the shared 160-byte result stream.

After a successful dispatch, one world generation makes stable CPU manifolds
lazy mirrors without writing a stale flag per contact. The CPU collision task
consumes compact exception records directly by contact ID. If there are no
exceptions, the task is not scheduled. Direct diagnostic calls retain the old
active-result behavior so the narrow-phase oracle tests can inspect every
supported output.

This removes the shared stable-manifold result stream and flat collision-worker
traversal. It does not yet remove CPU gathering of graph contact IDs, packing of
32-byte input records, or graph walks used to prove resident solver coverage and
schedule eligibility. Those walks are the next residency boundary.

## Correctness evidence

The 81-contact sphere/capsule differential seeded all contacts through the CPU
once, then ran three unchanged resident steps. It reported 243 device
preparation refreshes, 243 collision bypasses, 81 cumulative CPU collision
contacts, zero exceptions on the latest step, zero shared manifold bytes, and
zero manifold synchronizations.

Enabling hit events on one stable contact produced exactly one ordered CPU
exception and one 160-byte shared record while the other 80 contacts stayed
resident. Disabling that event and inserting one new touching contact again
produced exactly one first-touch exception and one 160-byte record. The
unchanged contact schedule packed once and reused three times before topology
insertion forced its deterministic rebuild.

Warm-start carry retained feature-matched persistence with one seed CPU
collision contact and zero latest exceptions. An unsupported-joint solver
fallback also retained one seed contact, zero latest exceptions, and
materialized its lazy manifold before CPU preparation. Float and double/VF64
warning-as-error Metal suites passed. The complete float debug,
AddressSanitizer, and UndefinedBehaviorSanitizer suites passed, as did the
focused double/VF64 UBSan Metal suite. The VF64 source was commit
`729021777455da72db8809d9ef1269c677d88b3f`.

## Structural benchmark evidence

The Release whole-world resident-contact harness used four substeps, eight
warmups, and 20 measured steps. Its new counters showed:

| Contacts | Collision bypasses | Cumulative CPU collision contacts | Latest exceptions | Latest shared manifold bytes |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 13,824 | 512 | 0 | 0 |
| 8,192 | 221,184 | 8,192 | 0 | 0 |

The CPU count is the seed step only; unchanged measured resident steps emitted
no shared manifold payload and processed no CPU collision contacts.

Timing from this run is deliberately not promoted. The host had load averages
above 80 with an `ffmpeg` process consuming roughly ten CPU cores, plus active
WindowServer and browser work. Raw samples varied by several multiples, so they
do not establish a CPU/Metal comparison or a new crossover. A quiet-host rerun
is required before making an end-to-end performance claim.

## Next boundary

Build a revisioned resident registry for contact input and graph order. Reuse it
while shape/contact generations and constraint-graph revision are unchanged,
then move solver coverage and schedule eligibility to compact device metadata.
That removes the remaining CPU gather, 32-byte-per-contact pack, and graph
coverage walks before another whole-world benchmark.
