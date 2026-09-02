# Exact GPU joint-pair filtering

Date: 2026-09-03

Host: Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory

Build: Metal warning-as-error, Float and Double/VF64

## Boundary removed

The Metal pair planner no longer treats every body with a joint as a CPU-filter
exception. Joint topology and `collideConnected` changes revise a deduplicated
open-addressed hash set containing only body pairs blocked by at least one
non-colliding joint. The same canonical full-body-ID key, hash finalizer, and
linear probing are used by the CPU upload and Metal count/write traversals.

Blocked candidates are rejected before residual classification. Candidates on
bodies with unrelated joints, or pairs whose joints all allow collision, remain
ordinary GPU plans. Custom-filter callbacks and compound-child expansion remain
the only residual CPU filtering reasons. Allocation, body-ID, key, capacity, or
hash validation failure returns the entire pair pass to the CPU oracle.

## Correctness evidence

- The existing mixed ordinary/custom/compound/joint differential remains exact.
  Its CPU exception moves fall from four to three because the blocked joint pair
  is now rejected on-device. Toggling that joint to allow collision produces one
  direct GPU candidate and identical contact topology.
- A parallel-joint differential proves Box3D's existential rule: a true and a
  false joint between the same bodies remain blocked, while changing the final
  false joint to true restores exactly one contact.
- Further mutations cover two false parallel joints, toggling only one to true,
  destroying the sole remaining false joint, rebuilding the registry, and
  recreating the contact with exact CPU/Metal IDs, generations, body edges,
  solver-set order, and pair-set membership.
- The mutation sequence performs four revision-driven registry uploads and
  reports zero CPU-filter moves throughout the joint-only scene.

## Performance interpretation

This checkpoint removes a candidate-dependent CPU branch for articulated
worlds; it does not claim a new steady-state whole-world speedup. Pair discovery
is normally cold or mutation-driven, while persistent contacts dominate later
steps. The prior whole-world measurements remain the relevant end-to-end
baseline: 1.279x for 131,072 resident box contacts, 1.139x for 262,144 contacts,
and 2.550x for the 131,072-body unconstrained finalization/AABB path. The new
public `pairFilterRegistryUploadCount` separates cold joint-registry rebuilds
from shape-metadata uploads so future articulated benchmarks can identify the
exact route rather than infer it from timing.

A post-change Release smoke run of the unchanged 131,072 resident-box workload
measured 19.360521 ms CPU versus 14.315771 ms Metal, or 1.352x. Its cold planner
telemetry reported 131,072 direct candidates, zero CPU-filter moves, and one CPU
candidate-traversal bypass. This loaded-host observation is a regression guard,
not an attribution of the timing difference to the joint registry.
