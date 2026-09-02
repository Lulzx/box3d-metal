# GPU final pair planning and residual-filter compaction

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Release, Metal enabled, 8 Box3D workers, 4 substeps

## Boundary removed

Metal broad-phase records now form the final pair plan for ordinary candidates.
The GPU classifies each move and stably compacts only move indices that still
need body-joint checks, a custom filter callback, or compound-child traversal.
Ordinary ranges do not enter the CPU candidate filtering task and do not allocate
`b3MovePair` records. The deterministic serial commit reads each range in reverse
traversal order, matching Erin's existing prepend-list order exactly.

Contact ID allocation, generation reuse, primary shape ordering, body edge
lists, solver-set placement, pair-set insertion, event flags, and island topology
remain in `b3CreateContact` on the CPU. Candidate density beyond the historical
`16 * moveCount` staging bound retains the previous all-record CPU consumption
path rather than changing that behavior in this increment.

## Correctness evidence

- Exact CPU/Metal topology equality for 28 contacts from eight mutually
  overlapping bodies, including contact IDs, generations, canonical shape IDs,
  child indices, both body edges, body list heads/counts, solver-set order, and
  pair-set membership.
- Five deterministic randomized trials with 48 bodies each, mixed static,
  kinematic, and dynamic types, sensors, group filters, and disabled masks.
- A mixed plan with one ordinary range plus four residual-filter moves:
  accepted and rejected custom callbacks, a two-child compound, and a
  non-colliding joint.
- `collideConnected` mutation forces a pair-metadata refresh and produces exact
  CPU/Metal topology on the next pair pass.
- Existing dense-capacity fallback, existing-pair suppression, resident-move,
  mutation, and full end-to-end tests remain green.
- Float and Double/VF64 warning-as-error suites, Float and Double UBSan suites,
  Float ASan, the complete portable CPU suite, and the live Metal demo pass.

## Whole-world measurements

The resident box-contact benchmark performs cold contact creation during its
warmup and reports the resulting planner telemetry alongside steady whole-world
timings.

| Contacts | Repeats | CPU | Metal | Speedup | Direct planned candidates | CPU-filter moves |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 4 | 16.251833 ms | 12.709219 ms | 1.279x | 131,072 | 0 |
| 262,144 | 3 | 31.976707 ms | 28.065874 ms | 1.139x | 262,144 | 0 |

Both runs reported one CPU candidate-traversal bypass, zero collision
exceptions, and zero shape synchronizations. These speedups are loaded-host
observations, not an interleaved causal comparison against the preceding
commit; the telemetry is the direct proof that every cold ordinary pair avoided
the CPU filtering/allocation task.

The 131,072-body unconstrained shape/AABB run measured 7.992938 ms CPU versus
3.134089 ms Metal, or 2.550x. Its last resident broad-phase pass processed
63,025 moves, returned no candidates, and uploaded zero move-list bytes.

No PDF artifacts were created or modified for this increment.
