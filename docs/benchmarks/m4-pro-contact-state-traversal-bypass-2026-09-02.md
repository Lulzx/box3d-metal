# M4 Pro contact-state traversal bypass — 2026-09-02

## Scope

An unchanged supported resident contact phase now skips the remaining
capacity-linear CPU contact-state bookkeeping when Metal emits no exceptions:

- worker contact-state bitsets are not cleared;
- worker bitsets are not unioned;
- the serial set-bit topology/event traversal has zero blocks to visit.

The bitsets are scratch storage, not authority. Diagnostic SAT, manifold, and
recycling counters are reset and aggregated separately. A later CPU exception
or fallback clears every worker bitset to the current contact-ID capacity before
collision workers can set a bit, so old storage is unreachable.

## Differential evidence

The 81-contact resident differential contains a first-touch seed phase that
sets topology bits followed by three zero-exception resident phases. All three
bypassed contact-state traversal, the latest clear byte count was zero, and the
existing transform and velocity errors remained `2.69e-05` and `3.58e-07`.
Public SAT/cache and every manifold-count diagnostic bucket also matched the CPU
oracle after the bypassed phases.

Hit-event mutation and first-touch addition each forced CPU exception work and
reported a nonzero latest bitset-clear byte count. The body-index/fast-state
mutation fixture recorded three stable traversal bypasses, then a fast-body CPU
exception with a nonzero clear. This exercises both directions of the deferred
clear boundary.

The complete non-Metal Release, float Metal debug, AddressSanitizer, and
UndefinedBehaviorSanitizer suites passed. Float and double/VF64
warning-as-error focused suites passed, as did focused double/VF64 UBSan.

## Structural harness evidence

The loaded-host Release resident-contact harness reported:

| Contacts | Metal collision phases | State-walk bypasses | Latest state clear bytes | Latest CPU exceptions | Latest manifold bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 48 | 47 | 0 | 0 | 0 |
| 8,192 | 28 | 27 | 0 | 0 | 0 |

Each run cleared and processed state bits only for its first-touch seed phase.
Every later phase retained resident inputs, emitted no collision exception, and
skipped state-bitset work. Timing is not promoted: concurrent browser, Node,
Spotlight, and display processes kept host load near 8 and produced visibly
inconsistent wall-clock rows.

## Next boundary

The stable contact path no longer performs a CPU gather, input rewrite,
manifold readback, collision task, solver ownership walk, contact-state bitset
clear/union/traversal, contact-preparation walk, schedule repack, or impulse
store. A quiet-host profile is still required to distinguish fixed Metal
submission/synchronization cost from the remaining CPU graph scheduling,
island/sleep work, events, and unsupported paths.
