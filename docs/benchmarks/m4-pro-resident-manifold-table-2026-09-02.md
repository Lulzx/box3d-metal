# M4 Pro resident manifold-table checkpoint — 2026-09-02

This checkpoint establishes a stable device address for every supported convex
manifold. It is ownership and deterministic-addressing evidence, not an
accepted whole-world performance result.

## Ownership cut

The existing 16-byte narrow-phase input uses its former padding word for the
Box3D contact id. During active-result scatter, Metal writes the finalized
80-byte record both to the ordered compact CPU stream and to a private table at
`table[contactId]`. The resident copy sets its identity field to the contact id,
so input permutation cannot change the stored bytes.

The table grows geometrically to the world contact-slot count and stays private.
It adds no kernel, command buffer, wait, or steady shared-memory payload. An
explicit diagnostic/fallback API can blit the current successful table on
demand; ordinary simulation never calls it. Entries are consumed only when the
current input marks that contact eligible, so stale slots for unsupported or
destroyed contacts are not authoritative.

The public profile exposes `lastNarrowPhaseManifoldTableCount`, the number of
contact slots addressable after the last successful narrow-phase dispatch.

## Differential evidence

The float mixed fixture writes 62 supported records into a 65-slot private
table. A single explicit staging blit matches every resident record against its
compact counterpart byte-for-byte after normalizing the identity field. The
test then reverses all 65 input contacts, dispatches again, and proves all 62
contact-id-indexed records remain byte-identical despite the changed compact
order.

Direct world-oriented geometry error remains `1.79e-7`; end-to-end applied
error remains `2.33e-7`. The double/VF64 fixture writes ten supported records
into 13 contact slots at a `1e12` origin; direct error remains `1.79e-7` and
applied error remains `1.94e-7`.

Float and double warning-as-error focused Metal suites pass. The full CPU suite,
full AddressSanitizer suite with Metal-runtime leak detection disabled, full
float UndefinedBehaviorSanitizer suite, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

CPU workers still consume the compact stream and materialize `b3Manifold`
objects. The next cut is to prepare supported Metal contact constraints from
the resident table, while compacting only callbacks, events, touching-state
transitions, unsupported geometry, and other CPU exceptions.
