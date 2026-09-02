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
	uint64_t jointDispatchCount;
	uint64_t jointFallbackCount;
	uint64_t finalizationDispatchCount;
	uint64_t finalizationFallbackCount;
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
	int lastNarrowPhaseHullShapeCount;
	int lastNarrowPhaseUniqueHullCount;
	int lastNarrowPhaseResultCount;
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
