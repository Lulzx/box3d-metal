# Correctness and validation

## Validation model

The CPU implementation is never replaced as the oracle. The suite runs matching
CPU and Metal worlds from the same definitions, advances both for multiple
steps/substeps, and compares transformations plus linear/angular velocities.
Direct primitive tests separately cover packed state and integration kernels.

## Differential coverage

- Position integration across 16,384 randomized states and flags.
- Fused velocity/position integration across 8,192 bodies.
- Body-finalization arithmetic across 4,096 randomized states and all 25 result floats.
- Integrated unconstrained worlds with 2,048 moving sphere, capsule, and hull AABBs.
- Exact filtered pair traversal over 607 mixed bodies and 620 proxies, including
  same-body overlaps, sensors, zero masks, and equal positive/negative groups.
  All 1,905 accepted candidates match CPU per-move count, offset, order, proxy
  id, tree type, shape id, query shape id, and six query fat-AABB bounds.
- Shape metadata uploads once, reuses unchanged state, and refreshes exactly
  once after a filter mutation; tree bounds refresh independently.
- Capacity growth requires one safe write retry; the same steady route then
  completes in one command buffer.
- A moving 2,048-body world reuses its resident tree with zero repeat uploads;
  every raw candidate still matches CPU traversal and order.
- A sparse 1,024-shape world compacts exactly 512 alternating enlarged shapes
  across four scan blocks; compact order and the resulting move array match the
  upstream shape order exactly.
- The same sparse world poisons all 512 moving CPU fat-AABB mirrors between
  steps and still emits the exact resident compact stream. A subsequent public
  body transform changes the tree revision and prevents stale resident reuse.
- Three sparse resident dispatches perform zero full shape-result applies. A
  deliberately corrupted CPU AABB is repaired by one selective 64-byte query
  readback, and disabling Metal synchronizes the remaining current generation.
- The 2,048-shape mixed world performs zero full applies and synchronizes exactly
  2,048 records through one body query plus the existing shape-query oracle.
- The ten-step contact world remains on the compatibility route and records ten
  full shape-result applies with zero selective synchronizations.
- The sparse resident world records one initial shape-input pack and one reuse;
  its transform mutation forces the second pack.
- A dedicated 64-shape world proves exact awake-order rejection: sleep and wake
  swap-removals force two rebuilds and synchronize 64 then 63 stale records.
  The following unchanged step reuses the rebuilt registry.
- A filter-only edit with immediate contact invocation disabled still invalidates
  cached no-contact eligibility and forces a rebuild plus compatibility apply.
- The ten-step convex contact world records one input pack and nine reuses.
- A ten-step contact world records one initial tree upload and ten device refits.
- At `(+1e8, -1e8)`, every VF64 AABB contains a fresh CPU oracle AABB computed
  from the same Metal-world transform; maximum inward error is zero.
- Disabling Metal broad phase rebuilds retained enlarged CPU nodes before the
  next CPU step.
- Dense pair candidate overflow with zero GPU dispatches and one CPU fallback.
- Convex friction, tangent velocity, twist friction, and rolling resistance.
- Convex restitution.
- Scalar multi-manifold mesh contacts.
- Forced contact graph overflow.
- Rigid, spring, limit, and motor distance-joint modes.
- Mixed distance joints and contacts.
- Mixed distance/parallel graph colors.
- Ordered mixed-joint overflow.
- Supported joints attached to static bodies.
- Unsupported revolute-joint fallback.

## Recorded error maxima

| Case | Maximum observed error |
| --- | ---: |
| Integrated unconstrained world | 1.19e-7 transform |
| Body-finalization arithmetic | 2.29e-5 across all result floats |
| Awake-shape AABBs | 3.81e-6 across all bound components |
| VF64 far-world AABB containment | zero inward error across 2,048 mixed shapes |
| Filtered pair candidates | 1,905/1,905 exact, including order |
| Distance joint plus contacts | 4.66e-10 |
| Convex friction contacts | 4.77e-7 transform, 3.98e-6 velocity |
| Convex restitution | 1.19e-7 transform, 2.38e-7 velocity |
| Mesh contacts | 4.77e-7 transform, 3.60e-6 velocity |
| Contact overflow | 2.38e-7 transform, 7.75e-7 velocity |
| Distance modes and overflow | 1.07e-6 transform, 8.27e-5 velocity |
| Mixed distance/parallel | 7.45e-9 transform, 1.79e-7 velocity |
| Mixed joint overflow | 1.82e-6 transform, 3.29e-4 velocity |
| Static-body joints | 4.47e-8 transform |

These are observed values from the recorded M4 Pro run, not universal error
bounds. Tests use explicit acceptance tolerances and are rerun in each supported
configuration.

## Build and runtime matrix

The completed acceptance matrix includes:

- Debug full unit suite with Metal enabled;
- Release full unit suite with Metal enabled;
- Release CPU-only full unit suite;
- double-precision Metal differential and full suites;
- AddressSanitizer full suite (`detect_leaks=0` on macOS);
- UndefinedBehaviorSanitizer full suite;
- warning-as-error Metal builds and focused suites in float and double modes;
- shared-library build and public demo runtime;
- CMake install audit including headers, dylib, and package configuration;
- `git diff --check`.

The private-result checkpoint reran the full CPU-only suite and focused float
and double warning-as-error Metal suites after its final source guard fix. The
full sanitizer and Metal matrices had already passed the same implementation.

The persistent-input checkpoint passes float/double warning-as-error Metal,
full CPU-only, full AddressSanitizer, full float UndefinedBehaviorSanitizer, and
double UndefinedBehaviorSanitizer Metal gates.

The resident pair-filter checkpoint reran those same gates. Double-precision
Metal continued through the pinned VF64 exact AABB boundary.

## What the tests do not prove

They do not prove bit-identical determinism, all future upstream revisions,
every possible world topology, automatic profitability, or that CPU-resident
pipeline stages have moved to the GPU. The compatibility table remains the
authoritative scope.
