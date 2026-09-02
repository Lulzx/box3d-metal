# M4 Pro body-property residency — 2026-09-02

This checkpoint removes the stable-step CPU pack of the 128-byte
`BodyProperties` record used by Metal integration. The finalization kernel now
writes its absolute quaternion and inverse world inertia back into the record
and clears force and torque. A world revision invalidates that device authority
when CPU APIs change a property or reorder the awake solver set.

## Correctness and invalidation

Both float and double/VF64 warning-as-error builds pass the complete
`MetalTest` suite. The 128-body convex-contact differential ran ten steps with
one property upload, nine reuses, and zero latest upload bytes. An external
linear-velocity mutation forced only the existing state upload and retained
property reuse. A subsequent force mutation forced exactly one new property
upload of `128 * 128 = 16384` bytes.

The revision is advanced by body create/destroy, awake-set wake/sleep and
enable/disable transfers, snapshot restore, explicit transforms, mass/inertia,
damping, gravity scale, force, torque, and wind. Unsupported routes and failed
commands clear resident authority.

## Whole-world structural evidence

Command:

```sh
BOX3D_METAL_FINALIZATION=1 \
BOX3D_METAL_BROAD_PHASE=1 \
BOX3D_METAL_SHAPES=1 \
BOX3D_METAL_WORLD_COUNT=512 \
BOX3D_METAL_WORLD_REPEATS=6 \
./build/metal-release/bin/metal_world_benchmark
```

The benchmark performs five warmups and six measured steps. The M4 Pro run
reported:

| Counter | Result |
|---|---:|
| body-property uploads / reuses / latest bytes | 1 / 10 / 0 |
| body-state uploads / reuses / latest bytes | 1 / 10 / 0 |
| finalization readback bypasses / latest bytes | 11 / 0 |
| finalization shape-walk bypasses | 11 |
| private move-event dispatches / syncs / latest bytes | 11 / 0 / 0 |
| transform device refreshes | 11 |
| pair fallbacks | 0 |

The observed 0.263x wall-clock result is not used as performance evidence: the
host was not established as quiet, and this 512-body run exists to prove
transfer/traversal structure. The remaining per-step CPU work still includes
the body finalization bookkeeping walk and solved-state readback/comparison.
