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
- Batched sphere/capsule/compact-hull geometry over 65 float contacts: 62 GPU
  records cover the prior sphere/capsule routes plus compact hull-sphere face,
  edge, corner, deep-overlap, and authoritative-separated cases. A compact
  speculative hull-sphere, a high-aspect hull-sphere, and a hull-hull record
  remain CPU-owned in the same batch. Direct CPU-oracle error is `1.79e-7`;
  end-to-end applied-manifold error is `2.34e-7`; point count, feature ids, and
  order match, and unchanged repeated output is byte-identical.
- The double/VF64 route covers ten GPU records out of 13 contacts at a `1e12`
  origin with the same fallback classes. Direct error is `1.79e-7` and
  applied-manifold error is `2.19e-7`.
- Eight supported hull shapes sharing one box geometry produce one cold
  geometry upload, at least two stable reuses, and one unique hull. Replacing
  one shape with `b3Shape_SetHull` produces exactly the second upload and two
  unique hulls, proving fail-closed mutation rebuilding.
- Replacing the first sphere with `b3Shape_SetSphere` produces exactly the third
  geometry upload, proving the generalized primitive registry rebuilds before
  the next 120-byte-input dispatch.
- The body-transform table records one cold upload and two same-step reuses.
  Hull/sphere mutation leaves it resident; `b3Body_SetTransform` forces exactly
  the second upload while geometry stays unchanged. Three solved steps advance
  it to four uploads at subsequent collision boundaries. The contact input is
  16 bytes.
- Integrated unconstrained worlds with 2,048 moving sphere, capsule, and hull AABBs.
- Exact filtered pair traversal over 607 mixed bodies and 620 proxies, including
  same-body overlaps, sensors, zero masks, and equal positive/negative groups.
  All 1,905 accepted candidates match CPU per-move count, offset, order, proxy
  id, tree type, shape id, query shape id, and six query fat-AABB bounds.
- Shape metadata uploads once, reuses unchanged state, and refreshes exactly
  once after a filter mutation; tree bounds refresh independently.
- A two-body lifecycle emits one new pair, suppresses it after contact creation,
  reuses the unchanged pair set without upload, and emits it again after contact
  destruction. Pair-set and shape-table revisions refresh independently.
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
- Resident preparation over 81 independent convex contacts: 64 sphere pairs
  plus 17 parallel capsule pairs, covering SIMD tails, two-point manifolds,
  friction, rolling resistance, tangent velocity, restitution, and warm start.
- Prepare-on-fallback recovery when a resident contact shares a world with an
  unsupported revolute joint.
- Pre-solve callback invocation excludes the contact from resident preparation
  authority without producing a fallback count.
- The 81-contact preparation fixture reports a 336-byte contact-ID schedule
  versus the former 12,096-byte padded metadata stream. Recycling, callbacks,
  and unsupported-joint preflight report zero submitted schedule bytes.
- The same fixture validates 6,480 bytes of generation-tagged impulse results
  versus 35,616 bytes of SIMD-wide records, with unchanged float and VF64 error
  maxima. Every resident lane resolves to a current contact-ID record.
- A resident sphere impact matches the CPU hit-event point, normal, and 10 m/s
  approach speed while reporting one 80-byte result.
- Four stable 81-contact steps perform one contact-schedule pack and three
  reuses. Adding an 82nd touching contact forces exactly the second pack while
  preserving the CPU/GPU velocity differential.
- A one-contact resident carry test validates result/contact generations and
  feature identity, rejects a deliberately mismatched contact generation, then
  poisons every CPU-mirror warm-start term. The next fresh supported step
  restores GPU-authored state and ends at `4.47e-08` velocity error with one
  schedule pack and one reuse.
- Four stable 81-contact steps report four all-contact store bypasses and zero
  event/public synchronizations. The hit-event oracle reports one bypass and
  exactly one exception sync. Contact, body, and shape data APIs plus recording
  snapshot capture independently restore a poisoned CPU mirror; a CPU fallback
  exposes no stale resident result.
- Float and double/VF64 manifold differentials validate Metal-authored
  COM-relative anchors, deterministic feature persistence, default material and
  tangent finalization, plus material-registry mutation. A separate fixture
  proves custom friction/restitution callbacks remain CPU-owned while rolling
  and tangent values come from Metal.
- Four stable 81-contact steps report 243 direct device refreshes of the
  preparation table: every contact on all three post-seed steps. A
  generation-guard fixture reports exactly one valid refresh after rejecting a
  deliberately stale contact generation, while a two-step custom-material
  fixture reports zero refreshes and retains its CPU callback values.

## Recorded error maxima

| Case | Maximum observed error |
| --- | ---: |
| Integrated unconstrained world | 1.19e-7 transform |
| Body-finalization arithmetic | 2.29e-5 across all result floats |
| Awake-shape AABBs | 3.81e-6 across all bound components |
| VF64 far-world AABB containment | zero inward error across 2,048 mixed shapes |
| Filtered pair candidates | 1,905/1,905 exact, including order |
| Sphere/capsule/compact-hull manifold geometry | 1.79e-7 float and VF64 double |
| End-to-end applied manifold | 2.34e-7 float; 2.19e-7 VF64 double |
| Resident convex preparation | 5.96e-8 transform, 3.58e-7 velocity float; 9.36e-8 transform, 3.58e-7 velocity VF64 double |
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

The existing-pair checkpoint again passed the complete sanitizer/CPU matrix and
focused float/double warning and VF64 gates after specializing CPU consumption.

The bounded hull-sphere checkpoint again passed full CPU-only,
AddressSanitizer, and float UndefinedBehaviorSanitizer suites, plus focused
float/double warning-as-error and double/VF64 UndefinedBehaviorSanitizer Metal
gates. The existing high-aspect ground restitution test remains unchanged and
passes through CPU GJK fallback.

The resident hull-geometry checkpoint reran that same complete matrix after
replacing per-contact geometry descriptors with a shader-visible persistent
registry. Float and VF64 oracle errors and byte-identical replay stayed
unchanged.

The resident shape-geometry checkpoint again passed the complete matrix after
moving sphere/capsule endpoints and radii out of the contact stream. Primitive
mutation forces exactly one rebuild; float and VF64 oracle errors remain
unchanged.

The resident body-transform checkpoint again passed the complete matrix after
moving float/VF64 transforms out of contact records. Teleport, solved-step, and
replay-seek invalidation are fail-closed; oracle errors remain unchanged.

The resident contact-preparation checkpoint passed the portable CPU suite,
float and double/VF64 warning-as-error Metal suites, full AddressSanitizer and
float UndefinedBehaviorSanitizer suites, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite.

The resident-schedule checkpoint reran the same matrix. Its 512-contact harness
smoke recorded one pack and nine reuses over ten successful dispatches.

The contact-prepare metadata-residency checkpoint reran the same matrix after
moving record writes into the existing parallel persistence pass and replacing
the solver-time contact walk with per-color bulk ID copies. Whole-world harness
smokes at 512 and 8,192 contacts each completed ten resident dispatches.

The contact-impulse residency checkpoint again passed the portable CPU full
suite, float and double/VF64 warning-as-error Metal suites, full AddressSanitizer
and float UndefinedBehaviorSanitizer suites, and focused double/VF64
UndefinedBehaviorSanitizer Metal suite.

The device contact-prepare refresh checkpoint passed the same matrix. Float and
double/VF64 differentials both reported 243 stable refreshes; CPU/GPU velocity
error remained `3.58e-07`, and the warm-start generation fixture remained
`4.47e-08`.

The contact-input residency checkpoint again passed the portable CPU suite,
complete float Metal debug, AddressSanitizer, and UndefinedBehaviorSanitizer
suites, float and double/VF64 warning-as-error builds and focused tests, plus
focused double/VF64 UBSan. A mutation fixture reused inputs across an awake-body
index swap, then forced exactly one CPU exception when the same body became
fast without changing the cache key. Broader gates also caught and fixed a
CPU-only conditional-compilation regression and a null cached-order prefetch.

## What the tests do not prove

They do not prove bit-identical determinism, all future upstream revisions,
every possible world topology, automatic profitability, or that CPU-resident
pipeline stages have moved to the GPU. The compatibility table remains the
authoritative scope.
