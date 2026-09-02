# M4 Pro resident existing-pair suppression — 2026-09-02

## Scope

This checkpoint mirrors Erin Catto's open-addressed contact pair set into a
persistent Metal buffer. The shader uses the same 64-bit shape-pair key, Murmur
finalizer, power-of-two mask, and linear probing as `b3ContainsKey`. Contact
creation, destruction, and snapshot restoration advance a revision; unchanged
sets stay resident without another copy.

Non-compound existing contacts are rejected during both the candidate count and
write traversals. Compound parents deliberately bypass this test because their
contact keys include child indices; they retain the complete CPU child query.
For ordinary candidates the CPU consumer now starts at joint/custom filtering
and deterministic move-pair append instead of repeating GPU de-duplication,
pair-set, sensor, same-body, and built-in filter work.

## Correctness gate

A dedicated two-body lifecycle test first observes exactly one candidate. After
the world creates the contact, the same moved proxies produce zero candidates
and one pair-set upload. An unchanged repeat produces zero candidates without
another upload. Destroying the contact through filter mutation, restoring the
filter, and moving both proxies produces exactly one candidate again, with one
pair-set and one shape-metadata refresh.

The 620-proxy mixed differential continues to match all 1,905 CPU candidates in
exact per-move order. Existing distance-joint/contact and ten-step convex and
mesh worlds preserve their CPU differential tolerances. Full CPU-only,
AddressSanitizer (`detect_leaks=0`), and float UndefinedBehaviorSanitizer suites
passed. Focused float/double warning-as-error and double/VF64 UBSAN Metal suites
also passed.

## Performance status

No timing is accepted from this checkpoint. The development host remained
loaded by a virtual machine and other interactive workloads. The pair-set mirror
currently copies its full retained capacity after a contact-set revision; stable
contact worlds reuse it, while high-churn worlds need a future incremental
device update path before a performance claim is justified.

The next ownership boundary is contact creation and shape-specialized narrow
phase. Compound child traversal, joint overrides, and user callbacks remain
explicit CPU exception paths.
