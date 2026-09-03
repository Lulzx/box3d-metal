# Performance evidence

## Measurement policy

Performance claims name the chip, OS, compiler, baseline, build, worker count,
body/constraint count, substeps, and timing boundary. Cold initialization is
not mixed into warm steady-state step timing. Whole-world comparisons include
CPU preparation/finalization, command submission, synchronization, and
readback unless explicitly labeled a primitive.

## Test platform

- MacBook Pro `Mac16,8`
- Apple M4 Pro: 12 CPU cores and 16 GPU cores
- 24 GB unified memory
- macOS 26.7 (`25G227`)
- Metal 4 compiler `32023.883`
- AppleClang 21.0.0, Release, `-ffp-contract=off`
- Eight Box3D workers in whole-world benchmarks
- Four substeps

## Summary

| Workload | Crossover/evidence | Largest measured speedup |
| --- | --- | ---: |
| Isolated position integration | Around 131,072 bodies | 1.278x at 524,288 |
| Fused four-substep integration primitive | Around 2,048 bodies | 10.584x at 524,288 |
| Unconstrained whole world | Only largest tested point | 1.146x at 524,288 |
| Convex-contact whole world | Around 262,144 bodies | 1.098x at 262,144 |
| Mesh-contact whole world | Around 8,192 bodies | 1.646x at 131,072 |
| Distance-joint whole world | Near 131,072 bodies | 1.158x at 524,288 |
| Parallel-joint whole world | No stable crossover | 0.974x at 1,048,576 median point |
| Experimental GPU finalization | No crossover | 27% slower at 524,288 bodies |
| GPU shape finalization | No crossover | 17.9% slower at 524,288 shapes |
| Experimental GPU tree traversal | Around 32,768 shapes | 1.068x at 524,288 shapes |
| Indexed cold-contact topology | Near crossover at 262,144 | 3.6% less GPU time than deferred-manifold checkpoint |

## Interpretation

Command fusion is the largest demonstrated architectural win: the fused
primitive reaches 10.584x, while full-world bookkeeping reduces that to 1.146x.
Mesh constraints cross earlier than convex-wide stacks in the tested setup.
Distance joints show a large-workload benefit despite compact pack/unpack costs.
Parallel joints expand compatibility but do not yet justify default routing.
The finalization arithmetic kernel is correct and fused into the resident
command graph, but its additional shared result stream made the 524,288-body
paired median 10.500 ms versus 8.275 ms with finalization disabled. It is
therefore an experimental opt-in, not a default optimization.

The follow-on shape kernel is also correct but does not yet change ownership:
at 524,288 bodies with one sphere each, the paired three-process median was
52.174 ms with body/shape finalization versus 44.235 ms with finalization off,
a 17.9% regression. The CPU still reads every 64-byte shape result, mutates the
dynamic tree, and traverses it for pairs. GPU pair generation must consume
resident bounds before this stage can remove that stream.

The first Metal dynamic-tree traversal slice produced an exact raw candidate
stream but copied the CPU tree, used an intermediate CPU prefix, and returned
candidates to CPU filtering. Against the same Metal finalization configuration,
that historical version improved the paired 524,288-shape median from 48.441 ms
to 45.356 ms (6.4%) while adding 86-107% below 2,048 shapes. Its two kernels used
about 3.0 ms at the largest point.

The current implementation replaces that CPU prefix with a deterministic
hierarchical Metal scan and compact write. A 607-proxy oracle compares all
8,081 candidates and their per-move order exactly; steady state uses one command
buffer. The follow-on resident-tree path updates enlarged leaves and refits
internal nodes on-device; a ten-step contact world uses one initial upload and
ten refits. VF64 also removes the double-precision shape fallback with
far-world CPU-oracle containment. No new whole-world number is accepted because concurrent CPU and MPS
loads contaminated the available host. The earlier crossover remains evidence
for the traversal architecture, not a benchmark of the current scan.

The resident contact-preparation checkpoint removes CPU preparation arithmetic
for complete supported colored convex sets and sources normal/identity from the
private contact-id table in the existing solver command buffer. It deliberately
adds no new whole-world timing from the loaded host. Its follow-on retains
post-persistence metadata by contact ID and replaces the dedicated contact walk
and padded 152-byte lane stream with a four-byte lane schedule. The 81-contact
fixture drops solver submission from 12,096 to 336 bytes. This remains ownership
evidence, not a loaded-host speedup claim.

The post-solve checkpoint adds an 80-byte result per active contact after
restitution. The same 81-contact fixture exposes 6,480 compact bytes instead of
21 complete 1,696-byte SIMD records, or 35,616 bytes. This 81.8% reduction is in
the CPU impulse-storage input surface; public-manifold and event synchronization
still runs on the CPU. Loaded-host smoke timing is not published as speedup.

The schedule-residency checkpoint retains the 336-byte lane schedule for stable
graph revisions. The 81-contact fixture performs one pack and three reuses;
adding one touching contact advances topology and forces exactly one repack.
A loaded-host 512-contact smoke recorded one pack and nine reuses across ten
dispatches. Its wall-clock values are not accepted as performance evidence.

The warm-start-carry checkpoint reuses padding in the 80-byte result for contact
generation and feature IDs, while the persistent preparation record grows from
144 to 152 bytes. A one-contact differential deliberately poisons the CPU
manifold mirror between steps; resident carry recovers the correct terms and
finishes at `4.47e-08` linear-velocity error with one schedule pack and one
reuse. This is correctness and ownership evidence, not a loaded-host speedup.

The lazy-sync checkpoint bypasses the all-contact CPU store after a successful
resident solve. Four stable 81-contact steps perform four bypasses and zero
event/public synchronizations; the hit-event differential performs one bypass
and exactly one exception sync. A loaded-host 512-contact smoke records ten
bypasses and zero syncs across ten dispatches. Its wall-clock timing is not
accepted as performance evidence.

The contact-finalization checkpoint moves feature persistence, both COM-relative
anchors, default material mixing, and tangent velocity into the existing
scatter. Its compact shared record grows from 80 to 160 bytes pending the next
exception-only topology path. A loaded-host 512-contact smoke measured `0.131x`;
that regression result is retained explicitly and is not crossover evidence.

The next checkpoint refreshes stable 152-byte preparation records directly in
the same scatter. Across eight warmups and 20 measured 512-contact steps it
reported 13,824 device refreshes, exactly all contacts on the 27 post-seed
steps. Whole-world time was `0.173540 ms` CPU versus `1.472546 ms` Metal
(`0.118x`). This is regression evidence, not acceleration: the compact
160-byte finalized-manifold stream and CPU contact/topology traversal remain.

GPU exception compaction removes that stable finalized-manifold stream and the
flat CPU collision task. The Release harness reported 13,824 bypasses for 512
contacts and 221,184 for 8,192 contacts; cumulative CPU collision counts were
only the 512 and 8,192 seed contacts. Both latest unchanged steps reported zero
exceptions and zero shared manifold bytes. These are structural counters, not
a speedup claim: the host load exceeded 80 with `ffmpeg` consuming roughly ten
CPU cores, and raw timings varied by several multiples. A quiet-host rerun is
required after the contact input/order registry work.

The contact-input residency checkpoint retains the exact 32-byte input/order
buffer across stable pair-set, graph, and eligibility revisions. Release smokes
at 512 and 8,192 contacts each recorded two packs, eleven reuses, zero latest
input bytes, zero latest exceptions, and solver coverage bypasses equal to all
stable collision bypasses (6,144 and 98,304). Only seed contacts reached CPU
collision. These are structural counters, not speedup evidence: a Python
process consumed one full CPU core and load averages remained about 4 to 8.
A quiet-host whole-world matrix is still required.

The zero-exception contact-state checkpoint also defers worker bitset clearing
until CPU collision work exists, and skips worker union plus serial set-bit
traversal on stable phases. Loaded-host runs recorded 47 bypasses across 48
Metal collision phases at 512 contacts and 27 across 28 at 8,192 contacts. Both
latest phases cleared zero state bytes and emitted zero exceptions. The same
loaded-host timing exclusion applies.

The empty-event solver checkpoint uses the current compact resident event-ID
count to defer all per-worker contact-capacity hit-bitset clears. The 512 and
8,192-contact harness rows bypassed all 48 and 28 resident solver clears and
reported zero latest hit-bitset bytes. Hit-enabled and forced Metal-fallback
fixtures restored a nonzero clear before CPU store work. Timing remains
excluded under load averages ranging from roughly 6 to 23.

The private-first-touch checkpoint replaces each ordinary cold contact's
240-byte shared manifold with a 16-byte ordered topology transition. At 131,072
contacts the clean pre-change GPU median of 115.063 ms falls to 93.112 ms
(-19.1%); at 262,144 it falls from 217.687 ms to 173.885 ms (-20.1%). The
corresponding median CPU/GPU ratios are 0.874x and 0.904x, so these are measured
regression reductions rather than crossover claims.

The deferred-manifold follow-on removes the remaining zeroed CPU allocation for
ordinary cold contacts. Against that private-first-touch checkpoint, the M4 Pro
GPU median falls from 93.112 to 86.685 ms at 131,072 contacts (-6.9%) and from
173.885 to 164.411 ms at 262,144 (-5.4%). The new CPU-oracle medians are 77.789
and 159.283 ms, leaving Metal 11.4% and 3.2% slower respectively. One large
sample crossed over; the three-run set supports only a smaller regression claim.

The indexed-topology follow-on replaces the 16-byte compact transition with an
8-byte contact-ID table and commits complete event-free batches directly in
canonical order. This removes per-worker state-bitset clearing, union, and the
second set-bit traversal. At 131,072 contacts the GPU median falls from 86.685
to 84.335 ms (-2.7%); at 262,144 it falls from 164.411 to 158.536 ms (-3.6%).
The remaining CPU island/graph commit takes median 7.995 and 15.504 ms,
respectively. One 262,144-contact trial crossed over, but the median is only
approximately parity, so this is not a universal crossover claim.

The retained-pair-seed follow-on removes the separate CPU-written 16-byte cold
input identity stream. Seven alternating same-host runs put the 131,072-contact
median at 80.775 ms versus 84.027 ms for `1e5f205`; at 262,144 contacts the
result is latency-neutral at 151.722 versus 150.498 ms, with a paired median
delta of only +0.116 ms. The defensible claim is transport ownership: 2 MiB and
4 MiB of CPU writes become zero, while Metal returns one 4-byte validation
status and retains the private 40-byte inputs.

The private-cold-topology epoch admits a strict dynamic-static virgin batch as
a device-private one-color schedule. Seven alternating runs against `efbdc6c`
put the cold step at 65.805 versus 83.317 ms at 131,072 contacts (paired delta
-18.644 ms) and 119.452 versus 153.860 ms at 262,144 (paired delta -32.280
ms), with zero transition bytes and zero direct commits. Deferred
materialization matches the eager CPU commit: 7.846 versus 7.944 ms and 15.196
versus 15.344 ms.

The data supports an explicit caller-selected threshold, not a universal
default. GPU frequency, CPU worker scheduling, constraint topology, contact
density, and unsupported stages can move the crossover substantially.

## Reproduce

```sh
./scripts/bootstrap.sh ../box3d-metal-worktree
./scripts/run-benchmarks.sh ../box3d-metal-worktree
BOX3D_METAL_SHAPES=1 ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_WORLD_COUNT=524288 BOX3D_METAL_WORLD_REPEATS=12 \
  BOX3D_METAL_SHAPES=1 BOX3D_METAL_FINALIZATION=1 BOX3D_METAL_BROAD_PHASE=1 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_world_benchmark
BOX3D_METAL_RESIDENT_CONTACT_COUNT=8192 BOX3D_METAL_RESIDENT_CONTACT_REPEATS=20 \
  ../box3d-metal-worktree/build/metal-release/bin/metal_resident_contact_benchmark
```

The benchmark script prints raw CSV-like rows. Run complete executables in at
least three separate processes and compare medians before selecting a threshold.
Detailed original tables are linked from [the documentation index](index.md).
