# M4 Pro resident box SAT cache — 2026-09-03

## Scope

Canonical box contacts now carry Box3D's SAT separation, feature type, and
feature indices through the Metal narrow phase. The first dispatch is seeded
from the CPU contact cache. Later dispatches read the prior private
contact-ID-indexed result table, so a shared contact-input repack does not erase
the cache. Cached face-A and face-B contacts rebuild the clipped manifold before
falling through to fresh SAT, matching Box3D's strict linear-slop acceptance.

The 240-byte result ABI is unchanged: two former padding words now hold the SAT
record. The shared contact input grows from 32 to 40 bytes. A repeated-dispatch
differential verifies at least one resident face-cache hit after deliberately
repacking that input, while manifold bytes remain deterministic after accounting
for Box3D's expected `hit` transition.

## Correctness boundaries

- Canonical boxes remain bounded to a 16:1 maximum/minimum extent ratio.
- Edge-pair cache records are emitted and retained, but currently run fresh edge
  SAT on the next dispatch instead of taking the cached-edge shortcut.
- An 80:1 floor experiment matched the cold manifold exactly but diverged after
  rotation because Metal and CPU disagreed on cached clipped-face acceptance.
  That aspect class therefore remains an explicit CPU exception.
- A recycled CPU contact now synchronizes its prior resident impulse record
  before CPU preparation consumes the recycled manifold.

## M4 Pro validation

The following passed on the local Apple M4 Pro:

- complete CPU release suite;
- complete Float Metal Werror suite;
- focused Double/VF64 Metal Werror suite;
- focused Float and Double/VF64 UndefinedBehaviorSanitizer Metal suites;
- focused AddressSanitizer Metal suite with leak detection disabled;
- 256-body Metal demo, 120 steps and four substeps, with zero solver fallbacks.

The whole-world convex-contact benchmark remains dominated by the known CPU
collision/finalization boundary. This run measured 0.896x at 131,072 bodies and
0.836x at 262,144 bodies. These are wall-clock values from the shared host and
are recorded as a regression guard, not a speedup claim. Moving the remaining
finalization and pair/contact topology work on-device remains the performance
path.

No PDF artifacts were regenerated or modified for this checkpoint.
