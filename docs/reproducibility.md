# Reproducibility and distribution

## Why an overlay repository

This repository intentionally does not vendor the Box3D codebase. It stores the
independent documentation, helper scripts, compact PDFs, and one source patch.
The patch applies only to the exact upstream revision in `manifest.env`.

Benefits:

- upstream authorship and history remain canonical;
- repository size stays small;
- the implementation delta is directly reviewable;
- reconstruction starts from a cryptographically identified Git commit;
- accidental drift to a different Box3D revision fails early.

## Reconstruction chain

1. Read `BOX3D_REPOSITORY`, `BOX3D_BASELINE`, and `BOX3D_METAL_PATCH` from
   `manifest.env`.
2. Clone the canonical repository without selecting a moving branch.
3. Check out the full baseline SHA in detached-head state.
4. Run `git apply --check` before modifying the checkout.
5. Apply the patch.
6. Configure with `BOX3D_METAL=ON`, unit tests and benchmarks enabled, and
   samples disabled.
7. Build and run the Metal differential suite and demo.

`scripts/verify-clean.sh` executes this chain in a fresh temporary directory.

## Repository contents

The public Git history contains no Box3D checkout, build directory, generated
object, Xcode cache, or benchmark executable. PDFs are committed because they
are requested distribution artifacts; their generator is committed alongside
them.

## Patch maintenance

The patch must not be silently retargeted to a newer Box3D revision. Updating
the baseline requires a fresh source audit, patch regeneration, full
differential matrix, and new benchmark provenance. Performance tables tied to
the old revision should remain dated rather than overwritten.

## Integrity checks

`scripts/verify-clean.sh` checks the patch, builds from the reconstructed tree,
runs focused Metal correctness tests, runs the public demo, and checks the
reconstructed worktree for whitespace errors. GitHub publication should occur
only after this succeeds.
