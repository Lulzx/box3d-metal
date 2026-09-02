# Resident warm-start carry (2026-09-02)

## Scope

This checkpoint makes the GPU-authored post-solve table authoritative for the
next fresh supported contact preparation. Its existing 80-byte result replaces
padding with contact-slot generation and two point feature IDs. The persistent
preparation record grows from 144 to 152 bytes to carry the current feature IDs
and contact generation into the extraction kernel.

During the next collision/persistence pass, Box3D matches current manifold
features against the prior resident result. Matching points recover normal
impulses; friction, twist, and rolling impulses recover at manifold scope before
the new preparation record is staged. Contact ID alone is insufficient:
generation matching prevents a destroyed/recreated contact slot from inheriting
stale solver state.

CPU manifold persistence and public impulse synchronization still execute at
this checkpoint. Resident carry means solver correctness no longer depends on
that public CPU mirror being current, which is required before the bulk
post-solve storage traversal can become lazy or exception-only.

## Evidence

A sphere-contact differential runs one CPU and one Metal step, verifies that
the resident feature and contact generation match the CPU manifold, then
deliberately poisons the Metal world's CPU manifold normal, friction, twist, and
rolling impulses with `1000`. The following fresh collision step recovers the
resident terms and finishes with `4.47e-08` linear-velocity error against the
untouched CPU oracle. The contact schedule performs one pack and one reuse.

The existing 81-contact sphere/capsule differential, hit-event comparison,
topology repack test, and float/double VF64 routes remain unchanged. No
loaded-host whole-world speedup is claimed.
