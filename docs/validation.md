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
- Exact raw pair traversal and three-block hierarchical scan over 607 mixed body
  types and 8,081 candidates, including per-move count, offset, order, proxy id,
  tree type, and shape id.
- Capacity growth requires one safe write retry; the same steady route then
  completes in one command buffer.
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
| Raw pair candidates | 8,081/8,081 exact, including order |
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
- double-precision Metal differential suite;
- AddressSanitizer full suite (`detect_leaks=0` on macOS);
- UndefinedBehaviorSanitizer full suite;
- warning-as-error Metal build and focused suite;
- shared-library build and public demo runtime;
- CMake install audit including headers, dylib, and package configuration;
- `git diff --check`.

## What the tests do not prove

They do not prove bit-identical determinism, all future upstream revisions,
every possible world topology, automatic profitability, or that CPU-resident
pipeline stages have moved to the GPU. The compatibility table remains the
authoritative scope.
