# M4 Pro canonical box contact residency — 2026-09-03

## Scope

This checkpoint adds a fail-closed hull-hull slice for equal canonical
`b3MakeBoxHull` pairs with exactly one static body. Metal now owns Erin's face
SAT, incident-face selection, feature-preserving clipping, deterministic
four-point reduction, Gauss-valid edge SAT, and edge-edge contact construction
for that slice. Half-edge and face-start topology is content-deduplicated in the
resident hull registry.

The four-point transport is end to end: the private manifold record is 240
bytes, the generation-tagged preparation record is 224 bytes, and the compact
post-solve impulse record is 112 bytes. Persistence validation and accounting
cover all four point bits. Unequal boxes, dynamic-dynamic pairs, arbitrary
hulls, meshes, callbacks, sleep, and CCD remain explicit CPU boundaries.

## Differential evidence

The direct narrow-phase differential includes axis-aligned and rotated boxes,
one four-point face manifold, and one Gauss-valid edge manifold. Point order,
features, normals, anchors, and separations are checked against
`b3CollideHulls`; the maximum error remained `1.79e-07` in the full float Metal
suite.

A separate 17-contact world exercises four consecutive complete steps. The
first step seeds Box3D's CPU topology; all following steps keep collision,
four-point persistence, preparation, solve, finalization, shape bounds, and
broad-phase traversal resident:

```text
resident collision bypasses       51 / 51
CPU collision contacts            17 cold, 0 steady
latest collision exceptions       0
shared manifold bytes             0 steady
four-point persistence matches    153 total
max transform error               1.19e-07
max velocity error                2.50e-06
```

## Whole-world signal

The Release harness ran independent static/dynamic canonical box contacts on
an Apple M4 Pro with four substeps and eight Box3D workers. These short loaded-
host samples are directional, not the pending quiet-host matrix:

| Contacts | Repeats | CPU | Metal | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4 | 0.248 ms | 2.077 ms | 0.119x |
| 8,192 | 2 | 0.875 ms | 1.219 ms | 0.717x |
| 32,768 | 2 | 3.532 ms | 2.927 ms | 1.207x |

At 32,768 contacts the measured steps reported zero collision exceptions, zero
shared manifold bytes, 294,912 resident collision bypasses, nine CPU body-walk
bypasses, and only the cold shape-result apply. Small box worlds remain firmly
CPU-faster because dispatch and synchronization overhead dominate.
