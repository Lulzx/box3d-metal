# M4 Pro resident pair filtering — 2026-09-02

## Scope

This checkpoint moves the first semantics-preserving broad-phase rejection
layer into the existing Metal dynamic-tree traversal. A per-node moved mark
table rejects self-pairs and applies Erin Catto's exact moved-proxy ordering
rule before candidates are counted. A revisioned 32-byte-per-shape table then
rejects same-body pairs, sensors, and the built-in group/category/mask filter.
Both count and write traversals use the same predicates, so the stable prefix
and output order are unchanged.

Shape metadata is packed only after shape creation, destruction, or filter
mutation. Unchanged steps reuse the resident table. Existing-contact
suppression, compound child traversal, joint collision overrides, custom user
callbacks, and deterministic contact creation remain in the complete CPU
callback, which also repeats the GPU predicates as a fail-closed oracle.

## Correctness gate

The differential traversal case contains 607 bodies and 620 proxies, including
same-body overlapping shapes, sensors, zero masks, and equal positive and
negative groups. Metal produced 1,905 candidates. Every per-move count, offset,
proxy id, tree type, and shape id matched the filtered CPU traversal in exact
order. The first call uploaded tree and shape metadata once; the unchanged
second call uploaded neither. A filter mutation refreshed metadata exactly
once, and a tree-bound mutation refreshed the tree exactly once. A following
resident traversal again matched the CPU oracle without another upload.

The focused float and double warning-as-error configurations passed. The full
CPU-only, AddressSanitizer (`detect_leaks=0`), and float
UndefinedBehaviorSanitizer suites passed, as did the focused double/VF64 UBSAN
suite. The pre-existing pathological-height test still rejects Metal before
dispatch and completes through one CPU fallback.

## VF64 boundary

No floating-point operation is added by this filtering layer. The double-world
shape/AABB path continues to use the exact IEEE-754 helpers vendored from
VF64-metal commit `729021777455da72db8809d9ef1269c677d88b3f`; pair filtering is
integer identity, bit-mask, and ordering logic over those resident bounds.

## Performance status

No whole-world timing is accepted for this checkpoint. During validation the
host had load averages above 3, with a virtual-machine process consuming about
one full CPU core and another Python workload consuming roughly one third of a
core. The 0.8 ms values printed by the correctness test are route diagnostics,
not benchmark evidence.

The next broad-phase ownership step is resident existing-pair suppression and a
specialized CPU consumer for the remaining compound/joint/custom cases. That
reduces the accepted stream before incremental contact creation and narrow
phase move on-device.
