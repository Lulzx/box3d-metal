# Attribution and scope

This repository distributes an independently maintained Metal overlay for
[Erin Catto's Box3D](https://github.com/erincatto/box3d). Box3D and the overlay
are MIT licensed. The bootstrap script clones Box3D from its canonical upstream
repository at the exact revision recorded in `manifest.env`, then applies the
overlay patch.

The double-precision Metal AABB boundary incorporates the exact integer
IEEE-754 implementation from
[VF64-metal](https://github.com/Lulzx/VF64-metal), pinned at commit
`729021777455da72db8809d9ef1269c677d88b3f` and distributed here under MIT.

This repository is not the canonical Box3D repository and is not an upstream
release. Its benchmark results apply only to the stated hardware, software,
revision, build configuration, and workloads.
