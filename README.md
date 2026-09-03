# Box3D

[![Build Status](https://github.com/erincatto/box3d/actions/workflows/build.yml/badge.svg)](https://github.com/erincatto/box3d/actions)
[![CLA assistant](https://cla-assistant.io/readme/badge/erincatto/box3d)](https://cla-assistant.io/erincatto/box3d)

![Box3D Logo](https://box2d.org/images/logo.svg)

Box3D is a 3D physics engine for games.

[![Introducing Box3D](https://img.youtube.com/vi/jr_Fzl2XwKU/maxresdefault.jpg)](https://www.youtube.com/watch?v=jr_Fzl2XwKU)

## Apple Silicon Metal backend

This fork adds an opt-in Metal compute backend while retaining the upstream CPU
implementation as its reference and fallback. It currently accelerates
position integration in constrained worlds and fuses velocity plus position
integration across every substep for worlds without active contacts or joints.
It also has GPU-resident colored convex and mesh contact solvers covering normal
impulses, friction, tangent velocity, twist friction, rolling resistance, and
restitution. Contact and joint graph overflow is solved in deterministic scalar
order inside dedicated Metal kernels. Distance joints (including springs,
limits, and motors) and parallel joints share the GPU-resident command graph;
other joint types fall back safely. Experimental, separately opt-in stages port
body/shape finalization and traverse Erin's existing broad-phase trees on Metal
with deterministic on-device candidate compaction. Self/moved-proxy duplicates,
same-body pairs, sensors, and built-in shape filters are rejected against a
persistent shape-metadata table. A revisioned mirror of Box3D's pair set also
suppresses existing non-compound contacts. Ordinary candidate ranges are final
GPU pair plans and bypass the CPU filtering/allocation task; serial contact
creation consumes them in Erin's exact deterministic order. Moves containing
compounds or custom callbacks are compacted on-device and retain the CPU filter
path. A revisioned device hash set rejects exactly the body pairs blocked by a
`collideConnected == false` joint; unrelated and collision-enabled joints stay
on the direct GPU plan. Zero-exception plans are flattened on-device into an
8-byte-per-contact seed stream in exact creation order, so the CPU does not walk
per-move query records or raw tree candidates. Their 52-byte query records,
16-byte candidates, and scan blocks now stay in private Metal storage. Worlds
containing compound or custom-filter shapes retain the shared residual path;
unusually dense direct plans preserve the prior 64-candidates-per-move capability
with an explicit private-to-shared materialization. Contact topology creation
remains CPU-owned. Tree snapshots now remain in device-private Metal storage;
a CPU tree revision stages and blits a fresh snapshot before traversal, while
unchanged and device-refit steps expose no shared authoritative tree copy.
ordinary awake-shape motion updates leaves and refits internal bounds on-device,
while topology changes and unsupported CPU mutations invalidate the snapshot.
Enlarged proxy bookkeeping remains in a private GPU-compacted stream rather
than rescanning every shape result or walking body shape lists again. The next
Metal pair query consumes that stream directly, marks moved leaves on-device,
and performs no CPU move-list upload; the CPU shape/tree oracle is restored only
at a query, mutation, route-change, or fallback boundary.
Double-precision worlds use VF64 software binary64 for exact world translation
and directed float AABB narrowing on Metal. Across unchanged resident steps,
the prior Metal fat bounds are now the authoritative containment input; any
CPU tree revision invalidates that state and reseeds it from the CPU oracle.
The full 64-byte shape result now lives in private Metal storage. Collision-free
non-CCD worlds with contact masks disabled, and stable fully resident convex-contact
worlds using the Metal broad phase, do not blit or apply that stream; public AABB
queries stage only the requested 64-byte record, while route changes and Metal
disable synchronize the remaining results explicitly.
Revision-stable steps also reuse the persistent 72-byte shape-input registry:
a monotonic awake-order revision protects each cached body index in constant
time, so stable steps neither scan body ids nor repack geometry, filters, proxy
keys, and local bounds. Sleep, wake, topology, transform, and filter changes
rebuild fail-closed.
On the sleep-disabled, non-CCD route, finalization also authors deterministic
move events into private Metal storage. Public event queries lazily materialize
only that 72-byte-per-body stream; unqueried steps perform no move-event
readback. VF64 builds preserve the exact device-authored binary64 translation
through this public boundary.
For unconstrained worlds and stable fully resident sphere/capsule/hull-sphere or
admitted canonical box-contact worlds on that same bounded route, the complete per-body CPU finalization
walk is omitted. Absolute transforms, centers, inverse inertia, cleared
forces/torques, transient flags, move events, shape bounds, and tree refit remain
authoritative in Metal storage across steps. Public transform or body-property
access, topology or eligibility mutation, a collision exception, recording,
unsupported constraints, sleep/CCD enablement, and Metal shutdown lazily
synchronize the CPU mirrors once. Profile counters distinguish traversal
bypasses from those explicit synchronization boundaries.
The shape-specialized narrow-phase route batches sphere-sphere, capsule-sphere,
capsule-capsule, bounded compact hull-sphere, and a fail-closed canonical
box-box subset in one Metal dispatch. The box path ports Erin's face SAT,
incident-face walk, feature-preserving clipping, deterministic four-point
reduction, Gauss-valid edge SAT, and edge-edge contact. It currently admits
canonical `b3MakeBoxHull` pairs, including dynamic-dynamic and unequal boxes,
when each hull's largest extent is at most 16 times its smallest extent;
high-aspect box pairs remain CPU-owned.
Supported spheres, capsules, and compact hulls live in a revisioned
Metal geometry registry: unchanged dispatches reuse primitive endpoints, radii,
hull points, planes, triangles, and shape descriptors, while geometry and
topology mutation rebuild fail-closed. Identical hull streams remain
content-deduplicated. A second body-id registry retains static and awake body
rotations, local centers, and VF64-capable world translations for the collision
step. Each 40-byte contact record carries eligibility, shape/contact identity,
contact generation, and a CPU-seeded SAT cache. Full 240-byte outputs stay in private Metal storage; a stable scan/prefix/scatter pass
returns only CPU exceptions, tagged by original contact index, in the same
command buffer. A complete ordinary cold bootstrap instead returns a 16-byte
identity/point-count transition while keeping geometry private. The scatter rotates normals into world axes, produces both
center-of-mass-relative anchors with VF64 translation subtraction, mixes default
friction/restitution/rolling parameters and tangent velocity, and feature-matches
warm starts against the prior GPU result. CPU workers consume exception results;
the compact cold path installs zeroed structural placeholders and preserves the
existing ascending contact-ID topology transition. The CPU retains manifold allocation, custom material callbacks,
pre-solve callbacks, event exceptions, and island mutation. Other shape pairs stay on the
unchanged CPU path. High-aspect and speculative hull-sphere contacts explicitly
retain CPU GJK. Double worlds use VF64 exact subtraction before narrowing
relative translations to the float convex-collision boundary.
The same scatter writes each active finalized record into a private table
indexed by Box3D contact id. This adds no steady-path readback or dispatch; explicit table
staging exists only for validation and fallback diagnostics.
When every colored convex contact remains authoritative after persistence and
callback processing, a Metal preparation kernel now builds Erin's SIMD-wide
contact constraints from that table in the solver command buffer. The CPU seeds
a 224-byte generation-tagged contact-ID table with body indices and manifold
identity. On later generation-stable steps, the manifold scatter refreshes that
record directly with current indices, finalized anchors and materials, prior
contact-scope impulses, persistence, and normal warm starts. Recycling,
pre-solve callbacks, and custom material callbacks remain CPU-written
exceptions.
Once an already-touching contact has a generation-current device-refreshed
preparation record, the collision worker bypasses CPU manifold application and
leaves the finalized manifold authoritative in the private contact-ID table.
The CPU manifold becomes a lazy mirror. Public/debug/snapshot access, sleep
transitions, Metal disable, and CPU solver fallback materialize required
geometry by contact ID and generation. Ordinary complete cold first-touch plans
use the compact transition path. Fast/CCD, hit-event, recording, callback,
sleeping-body, separated, and mixed topology-changing contacts stay on the CPU path.
Solver submission bulk-copies only a deterministic four-byte contact-ID schedule
per SIMD lane; it no longer walks and dereferences every contact to repack the
records. Normal and identity remain private on-device. Mixed, recycled, callback,
overflow, or unsupported solver worlds fail closed to CPU preparation, including
explicit prepare-on-fallback recovery when a later constraint rejects the route.
After the final restitution pass, Metal writes a 112-byte compact impulse record
per active contact into a generation-tagged contact-ID table. Successful
resident steps bypass the all-contact CPU impulse-store traversal. Hit-enabled
contact IDs are compacted while the existing narrow-phase input is packed and
only those exceptions synchronize before ordered event construction. Contact,
body, and shape queries synchronize requested manifolds on demand; force debug
drawing and snapshots are explicit synchronization boundaries. Invalid or
unsupported routes retain the original store path, and CPU fallback invalidates
prior GPU result authority.
The constraint graph carries a monotonic topology/order revision. The four-byte
contact-ID lane schedule remains in its Metal buffer while that revision and its
exact wide/contact counts are unchanged; contact or joint insertion/removal
invalidates it before the next solver submission.
The post-solve table also retains contact generation and per-point feature IDs.
On the next fresh supported collision pass, the scatter kernel matches features
and emits point persistence plus normal warm-start impulses; contact-scope
friction, twist, and rolling terms remain resident through preparation staging.
Contact-slot reuse cannot consume stale state.
The 81-contact differential now performs four store bypasses, bypasses 243
steady CPU manifold applications, and performs zero CPU manifold
synchronizations. A deterministic scan/prefix/scatter pass emits only CPU
exceptions: an unchanged resident step returns zero shared manifold bytes and
runs no CPU collision workers, while the hit-event and first-touch
differentials each return exactly one ordered 240-byte record. Contact mirrors
use a world generation instead of per-step stable-contact flag writes. A
revisioned contact input/order registry now retains the 40-byte records across
unchanged pair, graph, and eligibility revisions, so stable steps neither
gather graph contact IDs nor rewrite the input buffer. Current body indices and
fast-body flags come from the per-step body registry. Completely cold ordinary
pair plans now capture a 16-byte CPU-assigned contact identity during the
already-required deterministic topology commit. Metal expands that stream into
the authoritative private 40-byte input table, bypassing the cold graph/awake
contact gather and both CPU input scans. Hit/pre-solve events, callback, mixed, and unsupported
routes reject the bootstrap and retain the legacy pack. The same dispatch proves
complete convex graph ownership and skips the solver's per-contact coverage
walk. A zero-exception dispatch also bypasses capacity-linear contact-state
bitset clears, worker unions, and the serial state-change traversal; the next
exception or fallback clears before any CPU worker writes. CPU topology
mutation, callbacks, hit/pre-solve events, unsupported geometry, and non-ordinary cold/revision rebuilds
remain explicit fallback boundaries. The solver likewise defers its
contact-capacity hit-event bitset clears when the current resident compact event
list is empty, restoring them before an event-enabled path or Metal fallback.
This remains a residency checkpoint rather than a universal whole-world
speedup.
The stages remain off by default;
cold/topology shape-registry rebuilds remain CPU work, while GPU tree traversal
crosses over only in large measured worlds.
See [the architecture and compatibility contract](docs/metal_architecture.md)
for the exact supported surface and current CPU-only stages.

Build and validate on Apple Silicon:

```sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBOX3D_METAL=ON -DBOX3D_SAMPLES=OFF -DBOX3D_BENCHMARKS=ON
cmake --build build/metal-release
./build/metal-release/bin/test
./build/metal-release/bin/metal_demo
./build/metal-release/bin/metal_fused_benchmark
./build/metal-release/bin/metal_world_benchmark
./build/metal-release/bin/metal_contact_benchmark
./build/metal-release/bin/metal_resident_contact_benchmark
./build/metal-release/bin/metal_mesh_benchmark
./build/metal-release/bin/metal_joint_benchmark
./build/metal-release/bin/metal_parallel_joint_benchmark
```

Enable Metal per world after creation. The threshold is explicit because the
profitable crossover depends on the Apple GPU, CPU worker count, and workload:

```c
b3WorldId world = b3CreateWorld(&worldDef);
if (!b3World_EnableMetal(world, 32768)) {
    /* Metal was unavailable; the world remains on the CPU path. */
}

/* Research path: correct and fused, but not yet a measured whole-world win. */
b3World_SetMetalFinalization(world, true);
b3World_SetMetalBroadPhase(world, true);
```

Inspect `b3World_GetMetalProfile(world)` to verify device selection, dispatches,
fallbacks, and the most recent GPU time. Metal results are tolerance-equivalent,
not bit-identical to the cross-platform CPU path. Build with
`-DBOX3D_METAL=OFF` for the unchanged portable implementation.

## Features

### Collision

- Continuous collision detection
- Contact events
- Convex hulls, capsules, spheres, triangle meshes, and height fields
- Multiple shapes per body
- Collision filtering
- Ray casts, shape casts, and overlap queries
- Sensor system
- Character mover

### Physics

- Robust _Soft Step_ rigid body solver
- Continuous physics for fast translations and rotations
- Island based sleep
- Revolute, prismatic, distance, motor, weld, and wheel joints
- Joint limits, motors, springs, and friction
- Joint and contact forces
- Body movement events and sleep notification

### System

- Data-oriented design
- Written in portable C17
- Extensive multithreading and SIMD
- Optimized for large piles of bodies
- Cross platform determinism
- Recording and replay

### Samples

- Uses sokol to run with D3D11 on Windows, Metal on macOS, and OpenGL 4.5 on Linux.
- Graphical user interface with imgui.
- Many samples to demonstrate features and performance.

## Building all platforms

- Install [CMake](https://cmake.org/)
- Install [git](https://git-scm.com/)
- Ensure these run from the command line

## Building with CMake presets (recommended)

This uses the presets in `CMakePresets.json`.

- Windows: `cmake --preset windows` then `cmake --build --preset windows-release`
- Linux: `cmake --preset linux-release` then `cmake --build --preset linux-release`
- macOS: `cmake --preset macos` then `cmake --build --preset macos-release`
- Windows MinGW: `cmake --preset mingw-release` then `cmake --build --preset mingw-release`

Run the samples app (must be in the Box3D directory).

- Windows: `.\build\bin\Release\samples.exe`
- Linux: `./build/bin/samples`
- macOS: `./build/bin/Release/samples`

## Building for Visual Studio

- Install [Visual Studio](https://visualstudio.microsoft.com/)
- Run `build_vs2026.bat`
- Open and build `build/box3d.slnx`

## Building for Linux

- Run `build.sh` from a bash shell
- Results are in the build sub-folder

## Building for Xcode

- mkdir build
- cd build
- cmake -G Xcode ..
- Open `box3d.xcodeproj`
- Select the samples scheme
- Build and run the samples

## Building for Web

- [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
- `emcmake cmake -B build -DBOX3D_SAMPLES=OFF`
- `cmake --build build`

Box3D uses SSE2 with WebAssembly. Define `BOX3D_DISABLE_SIMD` to disable SSE2.

## Building and installing

- mkdir build
- cd build
- cmake ..
- cmake --build . --config Release
- cmake --install . (might need sudo)

## Using Box3D in your project

The core library has no dependencies beyond the C runtime (and `libm` on Unix). Linking it
gives you the `box3d::box3d` target.

I recommend to use FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(box3d
  GIT_REPOSITORY https://github.com/erincatto/box3d.git
  GIT_TAG v0.1.0)
FetchContent_MakeAvailable(box3d)

target_link_libraries(my_app PRIVATE box3d::box3d)
```

For a vendored copy or git submodule, point `add_subdirectory` at it:

```cmake
add_subdirectory(extern/box3d)

target_link_libraries(my_app PRIVATE box3d::box3d)
```

To use a copy installed with `cmake --install`, find the package:

```cmake
find_package(box3d 0.1 REQUIRED)

target_link_libraries(my_app PRIVATE box3d::box3d)
```

See [`docs/hello.md`](docs/hello.md) for a minimal first program.

## Compatibility

The Box3D library and samples build and run on Windows, Linux, and Mac.

You will need a compiler that supports C17 to build the Box3D library.

You will need a compiler that supports C++20 to build the samples.

Box3D uses SSE2 and Neon SIMD math to improve performance. SIMD can be disabled by defining `BOX3D_DISABLE_SIMD`.

## Documentation

The user manual lives in [`docs/`](docs/) and is built with Doxygen. Enable the `BOX3D_DOCS` CMake option and build the `doc` target.

## Community

- [Discord](https://discord.gg/NKYgCBP)

## Contributing

Pull requests are currently disabled. Instead, please file an issue for bugs or feature requests. For support, please visit the Discord server.

## Giving feedback

Please file an issue or start a chat on discord. You can also use [GitHub Discussions](https://github.com/erincatto/box3d/discussions).

## License

Box3D is developed by Erin Catto and uses the [MIT license](https://en.wikipedia.org/wiki/MIT_License).

## Sponsorship

Support development of Box3D through [Github Sponsors](https://github.com/sponsors/erincatto).

Please consider starring this repository and subscribing to my [YouTube channel](https://www.youtube.com/@erin_catto).

## LLM Usage

LLMs are used in the following areas:

- unit tests
- samples app
- migrating code between Box2D and Box3D
- build configuration
- code reviews
- benchmarking

Elsewhere all code is developed and written by me. I take responsibility for every line of code in Box2D/3D.
