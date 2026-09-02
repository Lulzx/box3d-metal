# M4 Pro awake-island bitset bypass — 2026-09-02

## Scope

Body finalization previously cleared one awake-island bitset per worker and
wrote an island bit for every awake body even when world sleeping was disabled.
The sleep stage is not entered in that mode, so none of that scratch state is
observable or consumed.

Sleep-disabled worlds now skip the bitset clear, per-body island lookup, island
bit write, and split-candidate bookkeeping. Sleep-enabled worlds retain the
original path. This does not remove the remaining CPU body finalization walk:
body transforms, move events, transient flags, force reset, and CCD decisions
are still materialized on the CPU.

## Differential evidence

The 81-contact CPU/Metal differential ran four sleep-disabled finalization
phases. All four reported `awakeIslandBitSetClearBypassCount`, with
`lastAwakeIslandBitSetBytes` equal to zero. Its existing transform and velocity
errors remained within tolerance.

A dedicated sleep-enabled fixture kept the body awake, recorded zero bypasses,
and cleared a nonzero eight-byte island bitset. This proves the original sleep
path remains active when its result is consumed.

The complete non-Metal Release, float Metal debug, AddressSanitizer, and
UndefinedBehaviorSanitizer suites passed. Float and double/VF64
warning-as-error focused suites passed, as did focused double/VF64 UBSan.

## Structural harness evidence

The loaded-host Release resident-contact harness reported:

| Contacts | Finalization phases | Awake-island clear bypasses | Latest clear bytes |
| ---: | ---: | ---: | ---: |
| 512 | 9 | 9 | 0 |
| 8,192 | 9 | 9 | 0 |

The rows are structural evidence only: a concurrent Python process held one CPU
core and the load average remained unsuitable for timing claims.

## Next boundary

The next larger finalization boundary is device authority for body transforms
and compact move events. That is required before the shared body-finalization
result stream and the remaining CPU body traversal can be removed safely.
