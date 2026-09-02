# Troubleshooting

## `b3World_EnableMetal` returns false

Confirm the process runs on macOS with an available Metal device and that the
library was built with `BOX3D_METAL=ON`. Inspect standard output for the Metal
library or pipeline compilation error emitted by Box3D.

## Dispatch counters remain zero

Check `minimumBodyCount`, awake body count, and whether the world has work for
that route. Unconstrained, contact, joint, and standalone position counters
measure different paths.

## Fallback counters increase

Compare the world against [the compatibility table](compatibility.md). Common
causes are unsupported joint types, reaction-threshold events, or unsupported
constraint overflow. A fallback is a correctness path, not a Metal failure.

## Metal is slower

This is expected below crossover or when CPU-resident preparation/finalization
dominates. Increase workload only if representative, use a Release build, run
separate process trials, compare medians, and include synchronization in timing.
Do not use GPU kernel time alone as a whole-world claim.

## Patch does not apply

Do not force it onto a different revision. Delete the incomplete destination
and rerun `scripts/bootstrap.sh`. Confirm the baseline in `manifest.env` and the
checked-out `HEAD` match exactly.

## Shared-library consumers cannot find Metal symbols

Build the library with `BOX3D_METAL=ON`. The public Metal header is included by
`box3d/box3d.h` only for consumers receiving that build definition from CMake.
Use the exported `box3d::box3d` target rather than manually assembling compile
definitions and framework links.
