# M4 Pro resident tree refit and VF64 AABBs — 2026-09-02

## Scope

This checkpoint removes the moving-world tree-upload loop from the supported
experimental Metal path. Shape finalization writes enlarged fat AABBs into the
resident dynamic-tree leaves. Separate dispatches refit internal nodes in
ascending stored height, so every child is complete before its parent and Erin
Catto's topology and depth-first traversal order remain unchanged. Worlds with
fast/CCD bodies or an invalid resident revision use the existing fallback.

Disabling Metal broad phase rebuilds enlarged CPU nodes before returning
ownership to the ordinary CPU path. This transition is covered by the debug and
UndefinedBehaviorSanitizer test configurations.

## VF64 double-precision boundary

Large-world Box3D keeps only absolute body translation in binary64. The Metal
shape kernel now carries the original body-center bits into VF64's exact
integer `ieee64` helpers, applies the same center-plus-delta-plus-origin sequence
as the CPU, adds each float local bound, and narrows lower/upper results with
the directed binary64-to-binary32 modes. The imported source is pinned to
VF64-metal commit `729021777455da72db8809d9ef1269c677d88b3f`.

The GPU's preceding float rotation can differ from the CPU by an FP32 rounding
step, so the double path applies a scale-aware 16-epsilon local arithmetic
envelope before exact translation. This is a deliberate conservative bound,
not a claim that Metal and CPU AABBs are bit-identical.

## Correctness evidence

- A 2,048-body sphere/capsule/offset-hull world at `(+1e8, -1e8)` dispatched
  Metal shape finalization with no double-precision fallback.
- Every GPU AABB contained a fresh CPU `b3ComputeFatShapeAABB` result computed
  from the same GPU-world body transform; maximum inward error was zero.
- The separate CPU-world differential had maximum AABB component error
  `2.03e-06` and maximum body-position error `8.94e-08`.
- Direct resident traversal after the moving step matched every CPU candidate
  count, proxy id, tree type, shape id, and order without uploading a tree.
- A ten-step contact world recorded one initial tree upload and ten device
  refits in both float and double builds.
- Full float and double test suites passed. Focused float ASAN/UBSAN, focused
  double UBSAN, and warning-as-error builds for both precision modes passed.

## Performance status

No whole-world timing is published for this checkpoint. `textunderstandingd`
and `mediaanalysisd` were concurrently consuming substantial CPU during the
validation window, so paired timing would not meet the repository's clean-host
gate. Earlier finalization and pair-generation tables remain historical
evidence for their recorded implementations, not measurements of resident
refit or VF64.

The major remaining ownership cost is the 64-byte-per-awake-shape shared result
stream consumed by CPU shape bookkeeping. Pair traversal no longer needs that
stream or a fresh CPU-tree upload on the supported path, but events, public
queries, CCD bookkeeping, and CPU fallback still do. Eliminating it requires a
resident authoritative shape-bound store plus selective CPU synchronization,
not simply deleting the readback.
