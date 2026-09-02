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
suppresses existing non-compound contacts; joint overrides, compounds, custom
callbacks, and contact creation retain the CPU path. Tree snapshots remain in persistent Metal storage;
ordinary awake-shape motion updates leaves and refits internal bounds on-device,
while topology changes and unsupported CPU mutations invalidate the snapshot.
Enlarged proxy bookkeeping consumes a stable GPU-compacted subset rather than
rescanning every shape result or walking body shape lists again.
Double-precision worlds use VF64 software binary64 for exact world translation
and directed float AABB narrowing on Metal. Across unchanged resident steps,
the prior Metal fat bounds are now the authoritative containment input; any
CPU tree revision invalidates that state and reseeds it from the CPU oracle.
The full 64-byte shape result now lives in private Metal storage. Collision-free
non-CCD worlds with contact masks disabled do not blit or apply that stream;
public AABB queries stage only the requested 64-byte record, while route changes
and Metal disable synchronize the remaining results explicitly.
Revision-stable steps also reuse the persistent 72-byte shape-input registry:
an exact awake-body id sequence protects each cached body index, so geometry,
filters, proxy keys, and local bounds are not repacked or recomputed. Sleep,
wake, topology, transform, and filter changes rebuild fail-closed.
The shape-specialized narrow-phase route batches sphere-sphere, capsule-sphere,
capsule-capsule, and bounded compact hull-sphere manifold geometry in one Metal
dispatch. CPU workers consume the ordered local geometry
while retaining manifold allocation, warm-start feature matching, materials,
pre-solve callbacks, events, and island mutation. Other shape pairs stay on the
unchanged CPU path. High-aspect and speculative hull-sphere contacts explicitly
retain CPU GJK. Double worlds use VF64 exact subtraction before narrowing
relative translations to the float convex-collision boundary.
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
