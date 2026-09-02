# M4 Pro device contact-prepare refresh — 2026-09-02

## Scope

The manifold scatter now refreshes an existing 152-byte contact-preparation
record directly in shared Metal storage. It validates the prior impulse result's
contact ID, contact generation, and result generation, and separately validates
the retained preparation record's contact ID, contact generation, and manifold
identity. It then writes the current solver indices, preparation generation,
finalized default materials and tangent velocity, prior friction/twist/rolling
impulses, finalized COM-relative anchors, separations, feature IDs, persistence,
and normal warm starts.

The first eligible step still seeds the record on the CPU. Contact recycling,
pre-solve callbacks, and custom friction or restitution callbacks remain CPU
exceptions. The CPU also still consumes the 160-byte compact finalized-manifold
stream and runs contact allocation and topology processing. This checkpoint
therefore removes steady-state per-contact preparation-table writes, not the
shared result stream or collision traversal.

## Correctness evidence

An 81-contact sphere/capsule differential ran four steps with zero recycling.
Telemetry reported 243 device refreshes: every contact on each of the three
steps after the CPU seed. The float build's maximum transform and velocity
errors were `5.96e-08` and `3.58e-07`; double/VF64 reported `9.36e-08` and
`3.58e-07`.

The generation-guard fixture deliberately stages a record under a stale contact
generation, restores the valid generation, and then requires one device refresh
and the exact resident normal impulse. CPU/GPU velocity error remained
`4.47e-08`. A two-step custom-material fixture reported zero device refreshes
while retaining CPU callback values `0.123` friction and `0.456` restitution.

Float and double/VF64 Metal warning-as-error builds passed `MetalTest`. The
portable CPU suite, full float Metal AddressSanitizer and
UndefinedBehaviorSanitizer suites, and focused double/VF64 Metal UBSan suite all
passed on the Apple M4 Pro.

## Whole-world smoke

Hardware was an Apple M4 Pro with 12 CPU and 16 GPU cores and 24 GB unified
memory, running macOS 26.7 (25G227), Apple Metal 32023.883, and Apple clang
21.0.0. The Release benchmark used eight CPU workers, 512 resident sphere
contacts, four substeps, eight warmups, and 20 measured whole-world steps.

```text
CPU 0.173540 ms
Metal 1.472546 ms
speedup 0.118x
prepare dispatches 28
device prepare refreshes 13824
schedule packs/reuses 1/27
event/public syncs 0/0
```

The 13,824 refreshes equal 512 contacts across all 27 post-seed steps. The
timing remains a regression and is not evidence of acceleration: CPU manifold
application and traversal plus command-buffer latency still dominate. The next
structural checkpoint is to classify topology/callback exceptions so unchanged
resident contacts can bypass CPU manifold application, then eliminate the
compact shared result stream.
