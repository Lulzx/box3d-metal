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

## Inspect the route

```c
b3MetalProfile profile = b3World_GetMetalProfile(world);
printf("device=%s contacts=%llu joints=%llu contact_fallbacks=%llu\n",
       profile.deviceName,
       (unsigned long long)profile.contactDispatchCount,
       (unsigned long long)profile.jointDispatchCount,
       (unsigned long long)profile.contactFallbackCount);
```

Dispatch counts prove that a Metal stage ran. A zero fallback count proves only
that the recorded supported stages did not fall back; it is not a claim that
broad phase, narrow phase, CCD, sleeping, or finalization ran on the GPU.

## Disable and release

```c
b3World_DisableMetal(world);
```

The world remains valid and continues on the CPU path.
