// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "metal_timeline.h"

const char* b3MetalStageName( int stage )
{
	switch ( stage )
	{
		case b3_metalStageMutations:
			return "mutations";
		case b3_metalStageBroadPhase:
			return "broadPhase";
		case b3_metalStageNarrowPhase:
			return "narrowPhase";
		case b3_metalStageTopology:
			return "topology";
		case b3_metalStagePrepare:
			return "prepare";
		case b3_metalStageSolve:
			return "solve";
		case b3_metalStageFinalize:
			return "finalize";
		case b3_metalStageRefit:
			return "refit";
		case b3_metalStageEvents:
			return "events";
		default:
			return "unknown";
	}
}

uint64_t b3MetalAnalyticSolverBytes( int contactCount, int substepCount )
{
	if ( contactCount <= 0 || substepCount <= 0 )
	{
		return 0;
	}
	uint64_t passes = 1u + 3u * (uint64_t)substepCount;
	return (uint64_t)contactCount * 424u * passes;
}

int b3MetalExpectedSolveDispatchCount( int activeColorCount )
{
	if ( activeColorCount < 0 )
	{
		activeColorCount = 0;
	}
	return 10 + 13 * activeColorCount;
}
