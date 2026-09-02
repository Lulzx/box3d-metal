# Quick start

## Requirements

- Apple Silicon Mac
- macOS with the Metal framework
- Xcode command-line tools
- CMake 3.20 or newer
- Ninja
- Git

Confirm the essential tools:

```sh
uname -m
xcrun -sdk macosx metal -v
cmake --version
ninja --version
git --version
```

`uname -m` should report `arm64`.

## Reconstruct and build

```sh
git clone https://github.com/Lulzx/box3d-metal.git
cd box3d-metal
./scripts/bootstrap.sh ../box3d-metal-worktree
```

The destination must not already contain files. The script clones Box3D from
the canonical upstream URL, checks out the exact baseline, verifies the patch
before modifying the checkout, and creates `build/metal-release`.

## Validate

```sh
../box3d-metal-worktree/build/metal-release/bin/test
../box3d-metal-worktree/build/metal-release/bin/metal_demo
```

For a completely disposable reconstruction and focused acceptance run:

```sh
./scripts/verify-clean.sh
```

## Enable Metal

```c
b3WorldId world = b3CreateWorld(&worldDef);
if (!b3World_EnableMetal(world, 32768)) {
    /* Metal was unavailable. The CPU implementation remains usable. */
}
```

The threshold is a lower bound on awake body count, not an automatic hardware
recommendation. Use benchmark evidence as a starting point and measure the real
world configuration.

The experimental body and awake-shape finalization path requires a second
explicit opt-in. It is intended for pipeline research and is currently slower
end to end:

```c
b3World_SetMetalFinalization(world, true);
b3World_SetMetalBroadPhase(world, true);
```

The broad-phase opt-in currently moves raw dynamic-tree traversal and stable
candidate compaction. It is profitable only in the recorded large sparse
worlds and retains explicit CPU fallback for unsupported scan geometry, depth,
candidate-capacity, allocation, or dispatch limits.

## Inspect the route

```c
b3MetalProfile profile = b3World_GetMetalProfile(world);
printf("device=%s contacts=%llu joints=%llu contact_fallbacks=%llu\n",
       profile.deviceName,
       (unsigned long long)profile.contactDispatchCount,
       (unsigned long long)profile.jointDispatchCount,
       (unsigned long long)profile.contactFallbackCount);
printf("shape_geometry_uploads=%llu reuses=%llu hull_shapes=%d unique_hulls=%d\n",
       (unsigned long long)profile.narrowPhaseGeometryUploadCount,
       (unsigned long long)profile.narrowPhaseGeometryReuseCount,
       profile.lastNarrowPhaseHullShapeCount,
       profile.lastNarrowPhaseUniqueHullCount);
printf("transform_uploads=%llu reuses=%llu\n",
       (unsigned long long)profile.narrowPhaseTransformUploadCount,
       (unsigned long long)profile.narrowPhaseTransformReuseCount);
```

Dispatch counts prove that a Metal stage ran. A zero fallback count proves only
that the recorded supported stages did not fall back; it is not a claim that
broad-phase tree mutation, contact creation, every narrow-phase pair, CCD, or
sleeping ran on the GPU. The separately enabled finalization path has body and shape
dispatch/fallback counters. Pair traversal has independent pair dispatch,
fallback, and GPU-time fields.
The narrow-phase geometry counters distinguish a cold/rebuilt shape registry
from stable reuse and report supported hull-shape versus unique-hull counts.
Transform counters independently expose body-table rebuilds and same-step reuse.

## Disable and release

```c
b3World_DisableMetal(world);
```

The world remains valid and continues on the CPU path.
