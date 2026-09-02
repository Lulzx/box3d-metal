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
	int minimumBodyCount;
	uint64_t positionDispatchCount;
	uint64_t positionFallbackCount;
	uint64_t unconstrainedDispatchCount;
	uint64_t unconstrainedFallbackCount;
	uint64_t contactDispatchCount;
	uint64_t contactFallbackCount;
	uint64_t jointDispatchCount;
	uint64_t jointFallbackCount;
	uint64_t finalizationDispatchCount;
	uint64_t finalizationFallbackCount;
	uint64_t shapeDispatchCount;
	uint64_t shapeFallbackCount;
	double lastPositionGpuMilliseconds;
	double lastUnconstrainedGpuMilliseconds;
	double lastContactGpuMilliseconds;
	double lastJointGpuMilliseconds;
	double lastFinalizationGpuMilliseconds;
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

/// Disable Metal compute and release the world's GPU resources.
B3_API void b3World_DisableMetal( b3WorldId worldId );

/// Return current Metal configuration and dispatch telemetry.
B3_API b3MetalProfile b3World_GetMetalProfile( b3WorldId worldId );

/** @} */
