// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base.h"
#include "id.h"

#include <stdbool.h>
#include <stdint.h>

/** @defgroup metal Apple Metal compute backend
 *  Optional Apple Silicon acceleration. Include box3d/box3d.h and build with
 *  BOX3D_METAL=ON.
 *  @{ */

typedef struct b3MetalProfile
{
	bool enabled;
	bool finalizationEnabled;
	bool broadPhaseEnabled;
	int minimumBodyCount;
	uint64_t positionDispatchCount;
	uint64_t positionFallbackCount;
	uint64_t unconstrainedDispatchCount;
	uint64_t unconstrainedFallbackCount;
	uint64_t contactDispatchCount;
	uint64_t contactFallbackCount;
	/// World steps whose colored convex constraints were prepared on Metal from
	/// the resident contact-id table.
	uint64_t contactPrepareDispatchCount;
	/// Optimistic Metal preparation steps recovered by running CPU preparation
	/// before the CPU solver fallback.
	uint64_t contactPrepareFallbackCount;
	/// Bytes in the most recent deterministic contact-id lane schedule. The
	/// post-persistence metadata table is retained by contact id and is not
	/// repacked into this stream.
	uint64_t lastContactPrepareIndexBytes;
	/// Bytes Metal authored for active contact-id-indexed impulse results in the
	/// latest successful resident convex solve.
	uint64_t lastContactImpulseResultBytes;
	/// Contact-ID schedule uploads after a topology/order change.
	uint64_t contactSchedulePackCount;
	/// Successful solver steps that reused the resident contact-ID schedule.
	uint64_t contactScheduleReuseCount;
	/// Contact points whose warm-start impulse and persistence flag were matched
	/// by the Metal manifold-finalization kernel.
	uint64_t contactPersistenceMatchCount;
	/// Stable contact preparation records refreshed directly by the Metal
	/// manifold-finalization kernel instead of rewritten by a CPU worker.
	uint64_t contactPrepareDeviceRefreshCount;
	/// Stable touching contacts that bypassed CPU manifold application after
	/// their solver preparation record was refreshed on Metal.
	uint64_t contactCollisionBypassCount;
	/// Contacts processed by CPU collision workers. Resident steps contain only
	/// deterministic callback and topology exceptions.
	uint64_t contactCollisionCpuCount;
	/// CPU collision exception records emitted by the latest Metal narrow phase.
	uint64_t lastContactCollisionExceptionCount;
	/// Narrow-phase contact input registry rebuilds after contact/order or
	/// eligibility mutation.
	uint64_t contactInputPackCount;
	/// Narrow-phase steps that reused the resident contact input/order registry.
	uint64_t contactInputReuseCount;
	/// Bytes written into the 32-byte contact input stream on the latest step.
	/// Revision-stable reuse reports zero.
	uint64_t lastContactInputBytes;
	/// Per-contact resident-ownership checks skipped because the unchanged
	/// collision dispatch proved complete convex graph coverage.
	uint64_t contactCoverageBypassCount;
	/// Collision phases that emitted no CPU exceptions and therefore skipped
	/// contact-state bitset clearing, union, and serial traversal.
	uint64_t contactStateTraversalBypassCount;
	/// Contact-state bitset bytes cleared on the latest collision phase.
	/// A zero-exception resident phase reports zero.
	uint64_t lastContactStateBitSetBytes;
	/// Successful resident solver phases that skipped all per-worker hit-event
	/// bitset clears because the current compact event-ID list was empty.
	uint64_t contactHitEventBitSetClearBypassCount;
	/// Hit-event bitset bytes cleared on the latest solver phase. An empty
	/// compact resident event list reports zero unless Metal falls back.
	uint64_t lastContactHitEventBitSetBytes;
	/// Finalization phases that skipped all per-worker awake-island bitset
	/// clears and writes because world sleeping was disabled.
	uint64_t awakeIslandBitSetClearBypassCount;
	/// Awake-island bitset bytes cleared on the latest finalization phase.
	/// Sleep-disabled worlds report zero.
	uint64_t lastAwakeIslandBitSetBytes;
	/// Individual resident manifolds materialized into the CPU mirror on a
	/// public, debug, snapshot, route-change, or solver-fallback boundary.
	uint64_t contactManifoldSyncCount;
	/// Resident convex steps that skipped the all-contact CPU impulse-store walk.
	uint64_t contactImpulseStoreBypassCount;
	/// Hit-event exception contacts synchronized during the compact store path.
	uint64_t contactImpulseEventSyncCount;
	/// Total individual contacts synchronized into the CPU/public manifold on
	/// event, query, debug-draw, or snapshot boundaries.
	uint64_t contactImpulseSyncCount;
	uint64_t jointDispatchCount;
	uint64_t jointFallbackCount;
	uint64_t finalizationDispatchCount;
	uint64_t finalizationFallbackCount;
	/// Bytes explicitly copied from private device finalization authority into
	/// the latest CPU apply mirror. Zero on the bounded sleep-disabled, non-CCD
	/// route; other routes retain the checked mirror.
	uint64_t lastFinalizationReadbackBytes;
	/// Successful private-device finalization phases that omitted the CPU result
	/// mirror because sleeping and continuous collision were disabled.
	uint64_t finalizationReadbackBypassCount;
	/// Finalization phases that skipped the per-body CPU shape-list traversal
	/// because Metal shape results covered every awake shape.
	uint64_t finalizationShapeTraversalBypassCount;
	/// Complete per-body CPU finalization walks omitted because Metal owns every
	/// required output on the bounded unconstrained route.
	uint64_t finalizationBodyTraversalBypassCount;
	/// Finalization phases that authored a deterministic private move-event
	/// stream in awake-sim order.
	uint64_t bodyMoveEventDispatchCount;
	/// CPU per-body public event writes omitted because the private stream is
	/// authoritative until the application requests it.
	uint64_t bodyMoveEventCpuWriteBypassCount;
	/// Public event queries that materialized the latest private stream.
	uint64_t bodyMoveEventSyncCount;
	/// Bytes copied by the latest public move-event query. Zero while the
	/// application leaves the private stream unobserved.
	uint64_t lastBodyMoveEventReadbackBytes;
	uint64_t shapeDispatchCount;
	uint64_t shapeFallbackCount;
	uint64_t shapeCompactDispatchCount;
	uint64_t shapeBoundsResidentDispatchCount;
	uint64_t shapeInputPackCount;
	uint64_t shapeInputReuseCount;
	uint64_t shapeResultApplyCount;
	uint64_t shapeBoundsSyncCount;
	int lastShapeResultCount;
	int lastEnlargedShapeResultCount;
	uint64_t pairDispatchCount;
	uint64_t pairFallbackCount;
	uint64_t pairTreeUploadCount;
	uint64_t pairMetadataUploadCount;
	uint64_t pairSetUploadCount;
	uint64_t pairTreeRefitCount;
	uint64_t narrowPhaseDispatchCount;
	uint64_t narrowPhaseFallbackCount;
	uint64_t narrowPhaseGeometryUploadCount;
	uint64_t narrowPhaseGeometryReuseCount;
	uint64_t narrowPhaseTransformUploadCount;
	uint64_t narrowPhaseTransformReuseCount;
	/// Finalization phases that refreshed the next-step narrow-phase transform
	/// registry directly from private device results.
	uint64_t narrowPhaseTransformDeviceRefreshCount;
	/// Cross-step awake solver-state uploads and exact resident reuses.
	uint64_t bodyStateUploadCount;
	uint64_t bodyStateReuseCount;
	uint64_t lastBodyStateUploadBytes;
	/// Constant-time state-residency revision checks replacing the former
	/// whole-array CPU comparison.
	uint64_t bodyStateRevisionCheckCount;
	/// Full awake-state materializations at a CPU query, mutation, route-change,
	/// recording, or shutdown boundary.
	uint64_t bodyStateSyncCount;
	/// Solved-state bytes copied into the CPU mirror by the latest step or lazy
	/// synchronization boundary. Zero while resident state remains unobserved.
	uint64_t lastBodyStateReadbackBytes;
	/// Lazy full awake-body simulation mirror synchronizations and the number of
	/// body records updated by the latest boundary.
	uint64_t bodySimSyncCount;
	uint64_t lastBodySimSyncCount;
	/// Cross-step 128-byte awake body-property uploads and revision-authorized
	/// device reuses. Finalization writes the next-step quaternion, inverse
	/// world inertia, and cleared force/torque directly into this stream.
	uint64_t bodyPropertyUploadCount;
	uint64_t bodyPropertyReuseCount;
	uint64_t lastBodyPropertyUploadBytes;
	int lastNarrowPhaseHullShapeCount;
	int lastNarrowPhaseUniqueHullCount;
	int lastNarrowPhaseResultCount;
	/// Shared CPU-exception payload from the latest narrow phase. Stable
	/// resident steps report zero bytes.
	uint64_t lastNarrowPhaseResultBytes;
	int lastNarrowPhaseManifoldTableCount;
	/// GPU-authored convex contacts that survived callbacks/topology processing
	/// and entered the current solver graph.
	int lastResidentConvexContactCount;
	/// SIMD-wide constraint records covered by those contacts. This is non-zero
	/// only when every colored convex contact is resident-table authoritative.
	int lastResidentConvexConstraintCount;
	double lastPositionGpuMilliseconds;
	double lastUnconstrainedGpuMilliseconds;
	double lastContactGpuMilliseconds;
	double lastJointGpuMilliseconds;
	double lastFinalizationGpuMilliseconds;
	double lastPairGpuMilliseconds;
	double lastNarrowPhaseGpuMilliseconds;
	char deviceName[128];
} b3MetalProfile;

/// Enable Metal compute for a world. This is opt-in because GPU floating-point
/// evaluation is tolerance-equivalent, not bit-identical to the CPU path.
/// minimumBodyCount should be tuned for the host; values below 1 become 1.
/// Returns false if Metal initialization fails or the world is invalid/locked.
B3_API bool b3World_EnableMetal( b3WorldId worldId, int minimumBodyCount );

/// Opt into experimental GPU body-finalization arithmetic. This stage is kept
/// separate because current whole-world benchmarks do not yet show a stable
/// speedup; it is useful for correctness and pipeline-residency development.
B3_API bool b3World_SetMetalFinalization( b3WorldId worldId, bool enabled );

/// Opt into experimental GPU dynamic-tree traversal for broad-phase candidate
/// generation. Metal also performs deterministic moved-proxy de-duplication and
/// built-in same-body, sensor, and shape-filter rejection. A resident mirror of
/// the pair set suppresses existing non-compound contacts. Joint/custom filtering,
/// compounds, and deterministic contact creation retain the CPU path. Any
/// unsupported tree depth, capacity, or dispatch failure
/// falls back to the complete CPU traversal for that step.
B3_API bool b3World_SetMetalBroadPhase( b3WorldId worldId, bool enabled );

/// Disable Metal compute and release the world's GPU resources.
B3_API void b3World_DisableMetal( b3WorldId worldId );

/// Return current Metal configuration and dispatch telemetry.
B3_API b3MetalProfile b3World_GetMetalProfile( b3WorldId worldId );

/** @} */
