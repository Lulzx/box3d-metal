# M4 Pro private manifold-result checkpoint — 2026-09-02

This checkpoint removes the dense shared narrow-phase result stream. It is
device-ownership, deterministic compaction, and differential correctness
evidence, not an accepted whole-world performance result.

## Ownership cut

The 80-byte result for every contact input is now written to private Metal
storage. In the same command buffer, a deterministic three-kernel pass scans
`eligible`, prefixes one total per 256-record block, and scatters only active
results to a shared compact buffer. The formerly unused fourth header word now
stores the original `inputIndex`, so the record remains 80 bytes and carries its
stable place in the contact array.

No dense CPU lookup array replaces the old stream. Each Box3D collision worker
performs one lower-bound search for its input range and then advances linearly
through the ordered compact records. Unsupported shape pairs and the explicit
speculative hull-sphere fallback do not appear in the shared payload. The
manifold kernel plus scan, prefix, and scatter still use one command buffer and
one wait.

The public profile exposes `lastNarrowPhaseResultCount`, the number of records
made visible to the CPU for the last successful batch.

## Differential evidence

The float mixed batch has 65 contacts and produces 62 compact records, or
4,960 shared payload bytes instead of a dense 5,200-byte result. Their
`inputIndex` values are strictly increasing, repeated dispatches are
byte-identical, direct CPU-oracle error remains `1.79e-7`, and applied error
remains `2.34e-7`. The end-to-end collision path retains manifold persistence,
feature matching, callbacks, events, and graph/island transitions on the CPU.

The small byte reduction in this fixture is not a performance claim. The
structural gain is that shared traffic and CPU result traversal now scale with
active supported results rather than every awake contact in a mixed batch.

Float and double warning-as-error focused Metal suites pass on the M4 Pro. The
full CPU suite, full AddressSanitizer suite with Metal-runtime leak detection
disabled, full float UndefinedBehaviorSanitizer suite, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite also pass.

## Remaining ownership boundary

The CPU still validates and writes one 16-byte input per awake contact, waits
for narrow phase, and converts compact geometry into Box3D manifolds. Manifold
persistence, materials, callbacks, events, contact creation, and topology remain
CPU-owned. The follow-on
[`manifold-finalization checkpoint`](m4-pro-manifold-finalization-2026-09-02.md)
fuses world-axis orientation into the compact scatter.
