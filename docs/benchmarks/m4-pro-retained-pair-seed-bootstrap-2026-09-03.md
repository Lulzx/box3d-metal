# Retained pair-seed contact bootstrap on M4 Pro

Date: 2026-09-03

This checkpoint removes the separate CPU-written 16-byte cold contact identity
stream. A strict virgin pair plan retains the broad phase's existing 8-byte
shape-pair seeds. After CPU contact creation proves dense IDs, generation one,
canonical order, and unchanged revisions, a one-thread-per-contact Metal kernel
expands those seeds into the private 40-byte narrow-phase input table.

The route is opportunistic. Recycled IDs, callbacks, event flags, unsupported
geometry, residual CPU filtering, or any identity/revision mismatch cancel the
authority and use the checked legacy bootstrap or CPU pack.

## Device and method

- Apple M4 Pro, 12 CPU cores and 16 GPU cores, 24 GB unified memory
- macOS 26.7 (25G227)
- Apple clang 21.0.0 (clang-2100.1.1.101)
- Release, warnings as errors, eight Box3D workers, four substeps
- Whole-world cold sphere-contact step, wall-clock timing
- Baseline `1e5f205`; candidate working tree built from that exact commit
- Seven alternating baseline/candidate runs at each size

The process-level samples are noisy, so both raw medians and the median of the
seven paired candidate-minus-baseline deltas are reported.

| Contacts | Baseline median | Candidate median | Raw median change | Paired median delta |
|---:|---:|---:|---:|---:|
| 131,072 | 84.027 ms | 80.775 ms | -3.9% | -1.127 ms |
| 262,144 | 150.498 ms | 151.722 ms | +0.8% | +0.116 ms |

This is latency-neutral at the larger size and not claimed as a stable
whole-step speedup. It is a residency result: the candidate removes 2 MiB and
4 MiB of CPU-written bootstrap data respectively without adding another pair
record traversal.

## Transport proof

| Contacts | Existing shared pair seeds | Removed shared identity stream | Private input table | Shared status |
|---:|---:|---:|---:|---:|
| 131,072 | 1,048,576 B | 2,097,152 B | 5,242,880 B | 4 B |
| 262,144 | 2,097,152 B | 4,194,304 B | 10,485,760 B | 4 B |

The public profile reports one cold bootstrap dispatch, zero
`lastContactInputBootstrapBytes`, four
`lastContactInputBootstrapStatusBytes`, and `40 * contactCount` private input
bytes. The recycled-ID differential still reports the legacy `16 * count`
shared stream and exact LIFO-input to ascending-graph ordering.

Pair GPU medians were effectively unchanged at 131,072 contacts (7.856 versus
7.858 ms) and modestly lower at 262,144 (15.273 versus 14.881 ms). The rejected
prototype that authored 40-byte inputs during the pair traversal regressed the
whole cold step by about 5%; retaining seeds for narrow expansion avoids that
shape.

## Remaining bottleneck

The contact-ID-indexed first-touch topology table still returns 8 bytes per
contact, and CPU island/graph materialization still costs about 8.0 ms at
131,072 contacts and 15.5 ms at 262,144. The next meaningful cut is the bounded
private one-color cold topology epoch: consume the private schedule directly in
the solver, then materialize CPU topology only at an observation, mutation, or
fallback boundary.
