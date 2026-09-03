# Metal benchmark protocol (Phase 0)

All Metal numbers in `docs/benchmarks/m4-pro-*.md` must be collected under this
protocol so phases compare against the same baseline.

## Quiet host

- `caffeinate -dims` for the whole run (prevent sleep/display/idle).
- No browser, no video calls, no other builds running.
- `pmset -g therm` (or `sudo powermetrics --samplers thermal`) shows no thermal
  throttling before and after the run. If throttled, cool down and rerun.

## Repeats and reporting

- Every benchmark binary reports the **min and median of 5** timed steps after
  3 warmup steps (`metal_scene_benchmark` does this internally; pass custom
  scenes via `BOX3D_METAL_SCENE=<name>`).
- Accept a Metal number only if the **same-run CPU reference is within 5% of
  the stored CPU baseline** in the previous benchmark doc. Otherwise the host
  was noisy; rerun.

## What to record

- Apple chip, macOS version, Metal toolchain (`xcodebuild -version`), build
  type (`Release`), body/contact/joint counts, substep count, CPU worker
  count.
- Whether timing includes command submission, synchronization, and readback
  (wall-clock `b3World_Step` always does).
- Cold initialization (first step incl. PSO compile) separately from warm
  steady state.
- From `b3World_GetMetalProfile`: per-stage GPU ms, command-buffer count,
  dispatch count, barrier count, encode CPU ms, wait CPU ms, analytic solver
  bytes, achieved GB/s (bytes / wall time), and the `metal_bandwidth` probe
  result for the roofline denominator.

## Roofline

Run `tools/metal_roofline.py` with the scene's analytic solver bytes, step
ms, and probe bandwidth, and paste its output into the benchmark doc:

```sh
./build/metal-release/bin/metal_bandwidth
python3 tools/metal_roofline.py --contacts 65536 --bytes 23592960 \
  --step-ms 10.5 --bandwidth-gbs 200
```

## Signposts

Build with `-DBOX3D_SIGNPOSTS=ON` to emit `os_signpost` intervals (`step`,
`pairs`, `collide`, `solve`) for Instruments timelines. Default builds emit
nothing and pay nothing.
