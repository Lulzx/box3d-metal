# M4 Pro body-state revision gate — 2026-09-02

This checkpoint removes the whole-awake-array `memcmp` previously used to
decide whether Metal solver state could remain resident. Stable steps now use a
constant-time body count plus monotonic world revision check.

The revision advances for public velocity and target-velocity writes,
impulses, explosions, body flag and motion-lock changes, mass-center changes,
create/destroy, awake-set wake/sleep and enable/disable transfers, snapshot
restore, and every CPU solver/finalization route. A failed or unsupported Metal
command also clears the context's resident count.

## Differential evidence

The float warning-as-error `MetalTest` passes. Its 128-body convex-contact world
reports one upload and nine state reuses over ten stable steps, with ten
revision checks. `b3Body_SetLinearVelocity` advances the revision and the next
step performs exactly one full awake-state upload instead of reusing stale
device data.

## Whole-world structural evidence

The 512-body run used five warmups and six measured steps:

```sh
BOX3D_METAL_FINALIZATION=1 \
BOX3D_METAL_BROAD_PHASE=1 \
BOX3D_METAL_SHAPES=1 \
BOX3D_METAL_WORLD_COUNT=512 \
BOX3D_METAL_WORLD_REPEATS=6 \
./build/metal-release/bin/metal_world_benchmark
```

| Counter | Result |
|---|---:|
| body-state uploads / reuses | 1 / 10 |
| constant-time revision checks | 11 |
| latest body-state upload bytes | 0 |
| latest solved-state readback bytes | 28,672 |
| body-property uploads / reuses / latest bytes | 1 / 10 / 0 |
| finalization readback bytes | 0 |
| move-event readback bytes | 0 |
| pair fallbacks | 0 |

The readback value is exactly `512 * sizeof(b3BodyState)` (`512 * 56`). It is
reported deliberately: the next residency checkpoint must make public state
queries and mutations synchronize lazily before this final per-step copy can be
removed. The observed 0.232x timing is not performance evidence because this
was a structural run on a host not established as quiet.
