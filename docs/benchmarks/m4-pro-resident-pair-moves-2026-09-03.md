# Resident shape-finalization moves and Metal pair traversal

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Boundary removed

Successful shape finalization now retains the compact enlarged-proxy stream in
private Metal storage. The following Metal broad-phase dispatch consumes that
stream directly, marks moved leaves with epoch tags, and traverses the resident
tree without a CPU proxy-enlargement walk or move-list upload. Public queries,
CPU mutation, route changes, and failed dispatches conservatively synchronize
shape bounds and rebuild Erin's CPU move set before CPU fallback.

The profile counters used to prove the route are:

- `residentPairMoveDispatchCount`
- `enlargedShapeTraversalBypassCount`
- `lastPairMoveCount`
- `lastPairCandidateCount`
- `lastPairMoveUploadBytes`

## Correctness gates

The Float and Double Metal suites passed in warning-as-error builds. The Float
and Double UndefinedBehaviorSanitizer suites passed, as did the Float
AddressSanitizer suite with leak detection disabled. The unchanged CPU suite
also passed. These cover direct compaction/order checks, resident traversal,
mutation restoration, fallback, public AABB synchronization, and end-to-end
world differentials.

## Unconstrained whole world

Command:

```sh
BOX3D_METAL_FINALIZATION=1 \
BOX3D_METAL_BROAD_PHASE=1 \
BOX3D_METAL_SHAPES=1 \
BOX3D_METAL_WORLD_COUNT=131072 \
BOX3D_METAL_WORLD_REPEATS=8 \
build/metal-werror/bin/metal_world_benchmark
```

Five warmups preceded eight measured steps:

| Route | Whole-world time |
| --- | ---: |
| CPU | 7.833969 ms |
| Metal | 2.350484 ms |
| Speedup | 3.333x |

Metal kernel time was 1.958292 ms and pair-kernel time was 0.249000 ms. Seven
of eight measured pair phases consumed private resident moves. The last phase
processed 63,025 moves, returned zero candidates, and uploaded zero move-list
bytes. Thirteen shape-enlargement CPU traversals were bypassed across warmup and
measurement.

## Resident box-contact whole world

Current-host runs at 131,072 contacts measured 1.396x to 1.531x whole-world
speedup across three four-repeat samples. A 262,144-contact, three-repeat run
measured 44.669125 ms CPU versus 26.783529 ms Metal, or 1.668x. All runs reported
zero collision exceptions and zero shape synchronizations.

The final pair counters in this benchmark describe its cold contact-creation
phase, which necessarily starts from a CPU move list. A zero steady
`residentPairMoveDispatchCount` is therefore expected when the later settled
contact steps produce no enlarged proxies.

Earlier same-host measurements for the preceding SAT-cache increment were
0.896x at 131,072 contacts and 0.836x at 262,144 contacts. Those values establish
the observed regression that motivated this boundary removal, but they are not
a controlled side-by-side checkout comparison: CPU timings also moved between
runs. The defensible result is that current loaded-host runs crossed above 1x;
a strict causal speedup claim needs interleaved builds under identical load.

No PDF artifacts were created or modified for this increment.
