# Compact GPU contact-seed stream

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Boundary removed

For a zero-exception pair plan within Box3D's historical
`16 * moveCount` density bound, Metal now flattens candidate ranges into
`{shapeIndexA, shapeIndexB}` seeds. Seeds are eight bytes and are emitted in the
exact order previously reconstructed by two CPU loops: move index ascending and
candidate index descending inside each move.

The common CPU commit touches the tiny summary and flat seed stream. It no
longer scans every 52-byte query record to collect telemetry and then scans the
same records plus 16-byte raw candidates to create contacts. GPU summary fields
provide the direct and residual candidate totals. Custom callbacks, compound
children, dense plans, and dispatch failures retain the prior exact path or the
complete CPU oracle. At this checkpoint raw record and candidate buffers were
still shared GPU scratch. The subsequent private-scratch checkpoint removes
that remaining CPU-visible allocation for eligible ordinary worlds.

`b3CreateContact` remains serial and CPU-owned because it mutates the contact ID
pool and generations, two body edge lists, solver-set indices, the open-addressed
pair set, and several revision domains in deterministic order.

## Correctness evidence

- Every seed in the 620-move, 1,905-candidate traversal test is compared with
  the corresponding reverse-per-move raw candidate and query shape.
- The 28-contact all-overlap test consumes 224 seed bytes and retains exact
  contact IDs, generations, canonical shape IDs, both body edges, body heads and
  counts, solver-set order, and pair-set membership.
- Five randomized 48-body trials retain exact topology while taking the compact
  route.
- The mixed custom accept/reject, two-child compound, blocked-joint scene emits
  no seeds and preserves callback and child ordering on the residual path.
- Enabling collision on the joint produces one direct eight-byte seed and exact
  topology. Parallel-joint toggle and destruction tests remain exact.

## Whole-world measurements

| Phase | Contacts | CPU | Metal | Speedup | Moves | Seeds | Seed bytes | CPU-filter moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Steady | 131,072 | 16.896397 ms | 12.998052 ms | 1.300x | 262,144 | 131,072 | 1,048,576 | 0 |
| Steady | 262,144 | 31.172918 ms | 26.296057 ms | 1.185x | 524,288 | 262,144 | 2,097,152 | 0 |
| Cold first step | 131,072 | 113.660080 ms | 131.564255 ms | 0.864x | 262,144 | 131,072 | 1,048,576 | 0 |
| Cold first step | 262,144 | 229.535828 ms | 247.168411 ms | 0.929x | 524,288 | 262,144 | 2,097,152 | 0 |

For the two benchmark shapes per contact, the former CPU-read plan comprised
52 bytes per move plus 16 bytes per candidate: 15 MiB at 131,072 contacts and
30 MiB at 262,144. The new streams are 1 MiB and 2 MiB respectively, a 93.3%
reduction. Both runs report one record-traversal bypass and one seed dispatch.

The steady measurements remain GPU-faster, while the cold first step remains
CPU-faster because first-touch collision/manifold creation and contact topology
are still CPU-owned. These loaded-host timings are not a causal before/after
comparison; the byte counts and route counters are the direct evidence for the
removed traversal boundary.
