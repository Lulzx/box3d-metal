# Private Metal pair-traversal scratch

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Boundary removed

Ordinary worlds with no live compound or custom-filter shapes now allocate pair
query records, raw candidates, and prefix-scan blocks with
`MTLStorageModePrivate`. A compact plan exposes only the shared summary and the
8-byte-per-contact seed stream consumed by deterministic CPU contact creation.
The public profile reports one `pair_private_scratch_dispatches` event and
`last_pair_raw_shared_bytes=0` for this route.

For the benchmark's two moved shapes per contact, the private logical scratch is
`52 * moves + 16 * candidates + 16 * ceil(moves / 256)`: 15,745,024 bytes
(15.016 MiB) at 131,072 contacts and 31,490,048 bytes (30.031 MiB) at 262,144
contacts. The only pair-plan result range read by the CPU is respectively the
1 MiB or 2 MiB seed stream, plus the fixed summary. The reusable shared seed
buffer remains provisioned for the `16 * moveCount` capacity, so these figures
describe bytes produced and consumed, not total allocated shared capacity.

Eligibility is cached by shape revision and fails closed for the whole world.
Custom-filter and compound worlds retain the exact shared residual route. A
zero-exception plan above `16 * moveCount` retains Box3D's historical
`64 * moveCount` ceiling by explicitly blitting the private record and candidate
ranges to shared storage. Compound query shapes are rejected by a hard shader
gate before any partial plan is consumed.

## Correctness evidence

- The 620-move, 1,905-candidate traversal compares the private seed stream with
  Erin's CPU dynamic-tree query order without dereferencing raw GPU scratch.
- The exact 28-contact fixture and five randomized 48-body trials preserve
  contact IDs, generations, body-edge topology, solver-set order, and pair-set
  membership.
- The mixed custom-filter, compound-target, and blocked-joint scene remains on
  the shared residual route and preserves callback and compound-child order.
- A 20-candidate single-move fixture proves dense private-to-shared
  materialization reports 372 shared bytes, copies every candidate in CPU-tree
  order, and reproduces the CPU oracle's historical 16-contact fixed-capacity
  result exactly.
- A moved compound-query fixture proves the Metal shader rejects the request
  without exposing a partial plan. The existing validated CPU task does not
  admit compound queries, so this is a hard unsupported boundary rather than a
  claimed fallback path.

## Whole-world measurements

| Phase | Contacts | CPU | Metal | Speedup | Seeds | Seed bytes | Private dispatches | Raw shared bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Steady | 131,072 | 13.147597 ms | 10.304973 ms | 1.276x | 131,072 | 1,048,576 | 1 | 0 |
| Steady | 262,144 | 27.107237 ms | 23.828598 ms | 1.138x | 262,144 | 2,097,152 | 1 | 0 |
| Cold first step | 131,072 | 77.459206 ms | 117.299873 ms | 0.660x | 131,072 | 1,048,576 | 1 | 0 |
| Cold first step | 262,144 | 159.011169 ms | 223.790878 ms | 0.711x | 262,144 | 2,097,152 | 1 | 0 |

The steady path remains GPU-faster, but the cold first step remains decisively
CPU-faster because first-touch collision, manifold allocation, and deterministic
contact topology are still CPU-owned. These are loaded-host snapshots rather
than a controlled before/after timing comparison. The route counters and byte
telemetry are the direct evidence: the raw shared pair stream is zero on both
scales while exact topology and the 1/2 MiB seed boundary remain.
