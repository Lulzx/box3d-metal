# M4 Pro retained pair-seed bootstrap — 2026-09-03

This checkpoint removes the separate CPU-written 16-byte cold contact identity
stream. A strict virgin pair plan retains the broad phase's existing 8-byte
shape-pair seeds. After CPU contact creation proves dense IDs, generation one,
canonical order, and unchanged revisions, a one-thread-per-contact Metal kernel
expands those seeds into the private 40-byte narrow-phase input table.

Recycled IDs, callbacks, event flags, unsupported geometry, residual CPU
filtering, or any identity/revision mismatch cancel the authority and retain the
checked legacy bootstrap or CPU pack.

## Device and method

- Apple M4 Pro, 12 CPU cores, 16 GPU cores, 24 GB unified memory
- macOS 26.7 (25G227)
- Apple clang 21.0.0 (clang-2100.1.1.101)
- Release, warnings as errors, eight Box3D workers, four substeps
- Whole-world cold sphere-contact step, wall-clock timing
- Baseline source commit `1e5f205`; candidate source commit `efbdc6c`
- Seven alternating baseline/candidate runs at each size

Process-level samples are noisy, so raw medians and the median of the seven
paired candidate-minus-baseline deltas are both reported.

| Contacts | Baseline median | Candidate median | Raw median change | Paired median delta |
|---:|---:|---:|---:|---:|
| 131,072 | 84.027 ms | 80.775 ms | -3.9% | -1.127 ms |
| 262,144 | 150.498 ms | 151.722 ms | +0.8% | +0.116 ms |

The larger point is latency-neutral, so this is a residency claim rather than a
universal speedup. The candidate removes 2 MiB and 4 MiB of CPU-written
bootstrap data respectively without adding another pair-record traversal.

| Contacts | Existing shared pair seeds | Removed shared identity stream | Private input table | Shared status |
|---:|---:|---:|---:|---:|
| 131,072 | 1,048,576 B | 2,097,152 B | 5,242,880 B | 4 B |
| 262,144 | 2,097,152 B | 4,194,304 B | 10,485,760 B | 4 B |

The public profile reports one cold bootstrap dispatch, zero
`lastContactInputBootstrapBytes`, four
`lastContactInputBootstrapStatusBytes`, and `40 * contactCount` private bytes.
The recycled-ID differential still reports `16 * count` legacy bytes and exact
LIFO-input to ascending-graph ordering.

The contact-ID-indexed first-touch table still returns 8 bytes per contact, and
CPU island/graph materialization still costs about 8.0 ms at 131,072 contacts
and 15.5 ms at 262,144. The next cut is the bounded private one-color topology
epoch, with exact CPU materialization only at observation, mutation, or fallback
boundaries.
