# M4 Pro precompiled Metal library (Phase 5a)

## Host

- Apple M4 Pro, macOS 26.7, Xcode 26.6, `Release`, 8 CPU workers.
- No quiet-host protocol for this checkout run; absolute numbers are smoke
  validation, not the stored baseline (rerun under
  `docs/benchmarks/protocol.md` before comparing phases).

## Gate: world creation (EnableMetal) time

```
cold (archive miss, incl. populate+serialize):  94-132 ms
warm (blob + cached archive):                     ~3 ms
```

Warm creation went from ~300 ms of runtime shader compilation to ~3 ms,
100x under the 20 ms gate. Cold is dominated by first-run archive
population (~1.5 MB serialize); the blob load itself is milliseconds.

## What landed

- `src/metal/box3d_main.metal` + `src/metal/box3d_contact.metal`: the 39
  kernels as real MSL files, extracted byte-identically from the
  `metal_backend.m` string literals by `tools/extract_metal.py` (7
  layout-identical duplicate structs unified, zero kernel/helper collisions).
- `src/metal_abi.h`: shared header; `#if __METAL_VERSION__` selects the 68
  MSL struct layouts (topologically ordered) plus reserved function-constant
  indices (`B3_DOUBLE`, `B3_HALF_MATERIALS`, `B3_HULL_CLASS`,
  `B3_SUBSTEP_COUNT` — declared, no kernel reads them yet); `#else` carries
  the 75 C-side `_Static_assert`s moved verbatim from `metal_backend.m`.
- Build: `xcrun -sdk macosx metal -O2` compiles main+contact (+VF64 with
  `-DB3_DOUBLE_PRECISION` for double builds) into one
  `box3d_metallib.metallib` (~577 KB float), embedded via
  `tools/embed_metal.cmake` as a byte blob loaded with `newLibraryWithData:`.
  The two historical libraries merged into one (duplicate structs unified in
  the ABI header); all 39 kernel names verified present in the metallib.
- `BOX3D_METAL_RUNTIME_COMPILE=ON` (default) keeps `newLibraryWithSource:`
  from `metal_sources.h`, regenerated from the same `.metal` files (ABI
  header inlined; the contact part drops its include so the merged source has
  exactly one copy). `BOX3D_METAL_FORCE_SOURCE=1` forces the fallback path.
  Blob-only (`=OFF`) builds and passes.
- `MTLBinaryArchive` at `~/Library/Caches/box3d/<device>-<fnvhash>.bin`
  (FNV-1a over the exact library bytes, so shader edits invalidate stale
  caches): populated with all 39 kernels after PSO creation, serialized
  best-effort, never fails context creation. Note: the archive is currently
  write-only telemetry — `MTLComputePipelineDescriptor.binaryArchives`
  consumption is left for a later phase; the blob path does not need it, and
  Metal already keeps its own per-app pipeline cache.
- `MetalLibraryLoadTest`: blob load, forced-source fallback, archive
  miss→hit transition, warm-creation gate proxy (< 1000 ms bound; actual
  ~3 ms), with a step on each path proving the pipelines execute.

No behavior change: full `box3d_test` suite green on float, double,
CPU-only, warnings-as-errors+signposts, and blob-only configurations.
