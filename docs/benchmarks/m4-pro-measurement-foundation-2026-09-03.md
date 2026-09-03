# M4 Pro measurement foundation (Phase 0)

## Host

- Apple M4 Pro (12 CPU cores, 16 GPU cores, 24 GB unified), macOS 26.7,
  Xcode 26.6 (Build 17F113), `Release`, substeps 4, 8 CPU workers.
- No quiet-host protocol was enforced for this checkout run (browser open);
  treat absolute numbers as smoke validation of the harness, not the stored
  baseline. Rerun under `docs/benchmarks/protocol.md` before comparing phases.

## Bandwidth probe

```
bytes,repeats,median_gbps
268435456,20,102.07
# device=Apple M4 Pro
```

Achievable copy bandwidth ~= 102 GB/s. The plan's ~200 GB/s assumption was
optimistic for a blit round trip on this OS/toolchain; the roofline
denominator below uses the measured 102 GB/s.

## Scene harness smoke (`metal_scene_benchmark`, min+median of 5)

```
large_pyramid,resident=0,cpu,   min=2.28ms median=2.52ms
large_pyramid,resident=0,metal,  min=5.86ms median=6.16ms cmd=2 dispatches=131 barriers=130 encode=0.067ms wait=5.199ms narrow=0.248ms solve=4.602ms analytic=82,459,520B (13.4 GB/s) fallbacks=0
many_pyramids,resident=0,cpu,   min=2.90ms median=3.02ms
many_pyramids,resident=0,metal,  min=6.39ms median=6.58ms cmd=3 dispatches=131 barriers=130 encode=0.057ms wait=4.339ms narrow=0.294ms solve=3.338ms analytic=156,739,232B (23.8 GB/s) fallbacks=0
```

Read: both scenes still pay 4-5 ms of blocking wait across 2-3 command
buffers for ~0.3 + ~3-4.6 ms of GPU work; encode CPU is already negligible
(~0.06 ms). The Metal wall is ~2x the CPU wall on these scenes today, which
is the honest fallback/latency baseline Phase 1 (async, one command buffer,
zero blocking) must beat. Per-stage slots beyond pair/narrow/solve read 0
until those stages run on Metal.

## Roofline sample

```
python3 tools/metal_roofline.py --contacts 65536 --bytes 23592960 --step-ms 10.5 --bandwidth-gbs 102
contacts,bytes,bytes_per_contact,step_ms,floor_ms,achieved_gbs,roofline_gbs,fraction_of_roofline,headroom_vs_floor
65536,23592960,360.0,10.500,0.231,2.25,102.00,0.022,45.39x
```

## What landed

- `src/metal_timeline.h/.c`: `b3MetalStage` enum (9 stages), stage names,
  `b3MetalAnalyticSolverBytes` (424 B/contact/pass, passes = 1 + 3/substep),
  `b3MetalExpectedSolveDispatchCount` (10 + 13 * colors).
- `src/metal_signpost.h`: `os_signpost` intervals (`step`, `pairs`,
  `collide`, `solve`) under `-DBOX3D_SIGNPOSTS=ON`, no-ops otherwise.
- `b3MetalProfile`: `stageGpuMs[9]`, `lastCommandBufferCount`,
  `lastDispatchCount`, `lastBarrierCount`, `lastEncodeCpuMs`, `lastWaitCpuMs`,
  `lastAnalyticSolverBytes`, accumulated per step across pair/narrow/solve/
  finalize phases (GPU `GPUStartTime/GPUEndTime` per buffer; dispatch/barrier
  counts analytic until per-encode instrumentation).
- `MetalTimelineTest`: locks the 10 + 13 * colors formula plus an end-to-end
  unconstrained step asserting measured command buffers and encode/wait
  reporting.
- Harness: `metal_scene_benchmark` (7 scenes, CPU vs Metal), `metal_bandwidth`
  probe, `tools/metal_roofline.py`, `docs/benchmarks/protocol.md`.

No behavior change: full `box3d_test` suite green, CPU oracle untouched.
