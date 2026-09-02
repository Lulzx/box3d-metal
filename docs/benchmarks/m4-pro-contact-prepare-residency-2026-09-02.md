# Contact-prepare metadata residency (2026-09-02)

## Scope

This checkpoint removes the dedicated solver-time `b3Contact` traversal and
144-byte per-lane preparation stream introduced by the first Metal preparation
kernel. It does not move Box3D manifold persistence, materials, callbacks,
events, graph coloring, or impulse storage to the GPU.

During the already-required collision/persistence pass, each authoritative
contact originally wrote one 144-byte record into a persistent shared table
indexed by Box3D contact ID. The warm-start carry follow-on expands the current
record to 152 bytes with two point feature IDs and contact-slot generation.
Records carry a narrow-phase generation. Pre-solve callbacks, recycling, CPU
geometry, and stale generations cannot acquire solver authority.

Solver submission now clears tail lanes and bulk-copies each active color's
contact IDs into a four-byte-per-lane schedule. The backend performs at most one
bulk copy per active color and does not dereference `b3Contact`, `b3Manifold`, or
world contact storage. The preparation kernel validates the schedule against
both the generation-tagged metadata table and private manifold table.

## Evidence

The 81-contact differential fixture uses 21 four-lane constraints. Its current
solver schedule is 336 bytes (`21 * 4 * 4`). The former 144-byte-per-lane stream
included three padded tail lanes, so its allocation was 12,096 bytes
(`21 * 4 * 144`). This is a 97.2% reduction in the solver submission stream; it
is not a claim that persistence metadata disappeared.

Four consecutive four-substep steps produced four Metal preparation dispatches
with maximum float transform error `5.96e-08` and velocity error `3.58e-07`.
The recycling and pre-solve fixtures leave stale table records inaccessible and
report zero schedule bytes. The unsupported-joint fixture performs CPU
prepare-on-fallback before any schedule is submitted.

The profile field `lastContactPrepareIndexBytes` reports the current schedule
size and is zero on routes that never submit resident preparation.

Focused harness smoke runs at 512 and 8,192 contacts each recorded ten resident
preparation dispatches after eight warmups plus two measured steps. Their
schedule/prior-stream byte pairs were 2,048/73,728 and 32,768/1,179,648,
respectively. Wall-clock values from those loaded-host smoke runs are not
published as performance evidence.

No whole-world timing is claimed from the loaded development host. Follow-on
checkpoints now extract compact impulses, retain the graph schedule, and carry
warm-start state by feature ID. The lazy-sync follow-on also removes the
all-contact CPU store traversal. A quiet-host comparison remains next.

The reproducible whole-world harness is
`metal_resident_contact_benchmark`. It creates independent sphere contacts,
forces fresh supported narrow phase by disabling recycling in both worlds, and
prints CPU/GPU wall-clock step time, preparation dispatches, current schedule
bytes, and the equivalent prior padded-stream bytes. Environment variables
`BOX3D_METAL_RESIDENT_CONTACT_COUNT` and
`BOX3D_METAL_RESIDENT_CONTACT_REPEATS` select a focused point.
