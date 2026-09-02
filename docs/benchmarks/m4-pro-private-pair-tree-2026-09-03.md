# Device-private Metal pair tree

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Ownership boundary

The authoritative Metal snapshot of Erin's static, kinematic, and dynamic tree
nodes now uses `MTLStorageModePrivate`. When the CPU tree revision changes,
Box3D packs a shared staging buffer and encodes its blit before pair traversal
in the same command buffer. The device revision advances only after successful
completion. Unchanged pair queries reuse the private tree with zero staged
tree bytes, and shape finalization continues to update leaves and refit internal
bounds directly in that private snapshot.

The staging allocation remains CPU-visible and reusable for later topology
changes; this checkpoint changes authoritative ownership and steady access, not
total allocated unified-memory capacity. CPU mutation, route changes, and
fallback continue to invalidate or restore the CPU oracle before reuse.

## Correctness evidence

- The 620-move/1,905-candidate differential reports 70,560 logical private
  tree bytes on its first query, zero upload bytes on its unchanged second
  query, and a complete restage after explicit tree enlargement.
- Every compact seed remains in Erin's exact move-ascending,
  candidate-descending creation order.
- Exact 28-contact topology and five randomized 48-body topology trials remain
  unchanged.
- Dense materialization, custom/compound residual filtering, blocked-joint
  filtering, resident refit, and CPU fallback tests remain enabled.

## Whole-world telemetry

| Contacts | Latest tree upload | Private tree snapshot | Raw shared pair result |
| ---: | ---: | ---: | ---: |
| 131,072 | 25,167,216 bytes | 25,167,216 bytes (24.001 MiB) | 0 bytes |
| 262,144 | 50,333,040 bytes | 50,333,040 bytes (48.001 MiB) | 0 bytes |

These benchmark worlds create their pairs on the first step and do not need a
later pair query, so the public `last_pair_tree_upload_bytes` value describes
that latest initial pair phase. The direct unchanged-query differential is the
evidence for the zero-byte reuse path.

Three five-repeat steady samples were also recorded for the private-tree build
and the immediately preceding shared-tree commit `def103c`:

| Contacts | Shared-tree Metal samples | Private-tree Metal samples |
| ---: | --- | --- |
| 131,072 | 10.240, 15.465, 13.641 ms | 10.350, 10.149, 10.961 ms |
| 262,144 | 21.676, 21.849, 21.292 ms | 21.638, 22.237, 22.130 ms |

The 131k baseline was visibly noisy and the 262k medians differ by about two
percent in the opposite direction, so these timings do not establish a causal
speedup. Current cold first-step observations were 97.354 ms CPU versus
133.330 ms Metal (0.730x) at 131,072 contacts and 172.927 ms versus 245.926 ms
(0.703x) at 262,144. The cold regression remains dominated by CPU contact
topology, repeated narrow-phase input packing, manifold allocation, and
first-touch island/graph work; private tree ownership removes a residency
boundary but does not claim to solve that later pipeline cost.
