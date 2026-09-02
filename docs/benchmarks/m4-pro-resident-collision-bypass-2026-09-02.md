# M4 Pro resident collision-application bypass — 2026-09-02

## Scope

Stable touching convex contacts whose current preparation record was refreshed
by Metal now bypass CPU manifold application. Their finalized geometry remains
authoritative in the private, contact-ID-indexed 160-byte table; the existing
CPU manifold is marked as a lazy mirror. The generation tag occupies the
formerly unused fourth normal-header word, preserving the table ABI size.

The bypass is deliberately narrow. First touches, separations, fast/CCD
contacts, hit-event contacts, recording, pre-solve and custom-material
callbacks, recycling, and topology changes retain Erin Catto's CPU path.
Public queries, force debug drawing, snapshots, sleep transitions, Metal
disable, and CPU solver fallback materialize stale geometry by contact ID and
generation. Unsupported solver fallback uses one bulk readback before CPU
preparation; a device failure recomputes remaining stale convex mirrors with
the CPU oracle.

This checkpoint does not yet compact exceptions on the GPU. The 160-byte
active-result stream is still copied through shared memory and the CPU still
walks all awake contacts. It removes stable manifold application, not those two
remaining costs.

## Correctness evidence

The 81-contact sphere/capsule differential ran four resident steps and reported
243 device preparation refreshes, 243 CPU collision-application bypasses, and
zero manifold materializations. Adding a new touching contact forced its
first-touch CPU path while all 81 stable contacts bypassed again.

The warm-start fixture left one CPU mirror stale, then required a public contact
query to materialize it. Contact generation, feature ID, geometry, and the
resident normal impulse matched; CPU/GPU velocity error was `4.47e-08`.

An unsupported revolute joint forced solver fallback after one collision
bypass. One bulk manifold synchronization completed before CPU contact
preparation, cleared the stale marker, and preserved a CPU/GPU velocity error of
`4.26e-08`. A separate transition fixture synchronized once before island sleep
and once before Metal disable.

Float and double/VF64 warning-as-error Metal suites passed. The complete float
debug, AddressSanitizer, and UndefinedBehaviorSanitizer suites passed, as did
the focused double/VF64 UBSan Metal suite. The exact VF64 source remains commit
`729021777455da72db8809d9ef1269c677d88b3f`.

## Whole-world smoke

Hardware was an Apple M4 Pro, running macOS 26.7 (25G227), Apple Metal
32023.883, and Apple clang 21.0.0. The Release harness used eight CPU workers,
512 independent resident sphere contacts, four substeps, eight warmups, and 20
measured whole-world steps per sample.

Five independent samples were:

```text
CPU ms:   0.108337, 0.108744, 0.114723, 0.112231, 0.108327
Metal ms: 1.448394, 1.542190, 1.651081, 1.660117, 1.428944
median:   0.108744 ms CPU, 1.542190 ms Metal, 0.071x
```

Every sample reported 28 preparation dispatches, 13,824 device preparation
refreshes, 13,824 collision-application bypasses, zero manifold syncs, one
schedule pack, and 27 schedule reuses. The result is still a regression and is
not evidence of acceleration. The Metal time is effectively unchanged from the
preceding device-refresh checkpoint because the shared finalized-manifold
stream, flat CPU collision traversal, command submission, and synchronization
still dominate. The next structural checkpoint is GPU compaction of only
callback/topology exceptions, followed by removal of the shared stable-contact
stream and CPU traversal.
