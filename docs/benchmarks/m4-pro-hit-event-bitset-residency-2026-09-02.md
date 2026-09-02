# M4 Pro hit-event bitset residency — 2026-09-02

## Scope

The CPU solver previously cleared one contact-ID-capacity hit-event bitset per
worker before every solve, even when a complete resident convex set had no
hit-enabled contacts. The current narrow-phase registry already retains a
compact list of contact IDs whose shapes request hit events. Solver setup now
uses that list as a fail-closed clear gate.

When the resident convex set is complete, mesh and overflow contact counts are
zero, and the compact event list is empty, the bitsets remain untouched. The
Metal store stage cannot set `hasHitEvents` in that case, so stale bits are
unreachable. A nonempty event list clears normally. If the Metal solver fails
after a clear was deferred, its orchestrator clears every worker bitset before
advancing CPU store stages.

## Differential and fallback evidence

The 81-contact differential ran four complete resident solver phases with four
clear bypasses and zero latest hit-bitset bytes. Enabling hit events on one
contact retained the ordered compact exception behavior, produced the same hit
as the CPU oracle, and cleared a nonzero bitset before the store stage. Disabling
the event again restored the bypass on the following first-touch step.

The unsupported-revolute fixture deliberately enters solver setup with a
complete resident convex contact, then rejects the Metal solver. It recorded no
successful clear bypass and a nonzero latest clear byte count before CPU
fallback. This proves the optimization does not rely on Metal dispatch intent.

The complete non-Metal Release, float Metal debug, AddressSanitizer, and
UndefinedBehaviorSanitizer suites passed. Focused float and double/VF64
warning-as-error suites passed, as did focused double/VF64 UBSan.

## Structural harness evidence

The loaded-host Release resident-contact harness reported:

| Contacts | Resident solver phases | Hit-bitset clear bypasses | Latest hit-bitset bytes | Event synchronizations |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 48 | 48 | 0 | 0 |
| 8,192 | 28 | 28 | 0 | 0 |

Timing is not promoted. During this run the load averages ranged from about 6
to 23, so the rows serve only as structural proof that every no-event resident
solver phase avoided the contact-capacity clear.

## Next boundary

The supported no-event contact path now avoids both collision-state and solver
hit-event bitset traffic. A quiet-host profile remains the boundary for choosing
between fixed command synchronization, solver graph/stage orchestration,
body/island finalization, and the remaining event/topology exceptions.
