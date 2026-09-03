# GPU-authored cold contact inputs

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Ownership boundary

The ordinary completely cold pair route still commits Box3D contact topology
serially on the CPU. That commit is authoritative for contact ID allocation,
slot generation, primary shape ordering, body-edge links, pair-set mutation,
and awake-set order. It now records one 16-byte identity per created contact.
Metal validates and expands those identities into the private 40-byte manifold
input table before narrow phase in the same command buffer.

This removes the cold `b3GatherAwakeContactIndices` allocation/traversal, the
CPU eligibility/counting scan, the CPU 40-byte input pack, and the body-sim
materialization needed only by that pack. Stable contacts continue to reuse the
revisioned table. Hit/pre-solve events, custom callbacks, recording, sleeping
or partial topology, and unsupported geometry fail closed to the prior CPU
path. Contact recycling preparation remains CPU-owned.

## Correctness evidence

- The exact 28-contact topology test records 448 shared bootstrap bytes and
  1,120 private input bytes, with zero legacy input packs.
- A three-contact recycle differential frees IDs in ascending order and then
  recreates them in the ID pool's `2,1,0` LIFO order. Every slot generation
  increments, CPU/GPU topology remains exact, and the bootstrap uses 48 shared
  bytes to author 120 private bytes.
- An unsupported-joint solver fallback retains the successfully authored
  narrow-phase inputs without changing CPU contact identity.
- A hit-event cold world rejects bootstrap, performs exactly one legacy
  40-byte input pack, and preserves the CPU/GPU hit event.
- Post-commit seed corruption is rejected by the device status word; a contact
  input revision mismatch is rejected before dispatch. Both cases clear result
  outputs and bootstrap/reuse authority, then recover through an explicit
  CPU-packed call without changing contact identity or topology.
- Float and Double/VF64 warning-as-error Metal suites pass.

## Same-host cold whole-world comparison

The baseline is the immediately preceding source commit `e416116`, built in a
clean detached worktree. Each entry below is one complete cold process/world;
the table reports all three Metal samples and their median.

| Contacts | Baseline Metal samples | Bootstrap Metal samples | Median change |
| ---: | --- | --- | ---: |
| 131,072 | 127.655, 136.213, 132.818 ms | 115.436, 127.023, 124.851 ms | -6.0% |
| 262,144 | 242.970, 242.301, 237.226 ms | 236.716, 228.413, 232.113 ms | -4.2% |

The paired CPU medians were effectively stable at 113.420 ms for the baseline
and 113.804 ms for the bootstrap build at 131,072 contacts, and 229.048 ms
versus 227.490 ms at 262,144 contacts. The bootstrap build's median per-process
step ratios were 0.896x and 0.971x respectively. These are short, noisy samples,
so the byte and dispatch counters are the causal evidence; timings are an
end-to-end signal, not a universal speedup claim.

| Contacts | Shared identity stream | Private input table | Shared first-touch manifold stream |
| ---: | ---: | ---: | ---: |
| 131,072 | 2,097,152 bytes | 5,242,880 bytes | 31,457,280 bytes |
| 262,144 | 4,194,304 bytes | 10,485,760 bytes | 62,914,560 bytes |

The remaining cold gap is now more sharply isolated: first-touch results still
return a 240-byte record per contact, CPU workers allocate/apply manifolds, and
the serial island/constraint-graph transition still follows. The next larger
residency cut is therefore first-touch manifold-table bootstrap plus a compact
CPU topology/state-transition stream, not another solver micro-kernel.
