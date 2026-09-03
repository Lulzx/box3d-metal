// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// Logical Metal step stages. These index b3MetalProfile::stageGpuMs and the
// world's metalStageGpuMilliseconds mirror. Order is fixed; append only.
typedef enum b3MetalStage
{
	b3_metalStageMutations = 0,
	b3_metalStageBroadPhase,
	b3_metalStageNarrowPhase,
	b3_metalStageTopology,
	b3_metalStagePrepare,
	b3_metalStageSolve,
	b3_metalStageFinalize,
	b3_metalStageRefit,
	b3_metalStageEvents,
	b3_metalStageCount
} b3MetalStage;

#ifdef __cplusplus
extern "C"
{
#endif

// Human-readable short name for a stage, e.g. "solve".
const char* b3MetalStageName( int stage );

// Analytic DRAM traffic estimate for the current SIMD-wide solver layout:
// 424 bytes per contact per pass (1,696-byte b3ContactConstraintWide per 4
// contacts), with passes = 1 prepare + 3 per substep (warm start, solve,
// relax) plus the restitution sweep folded into the per-substep cost.
// This is the roofline input, not a hardware counter: the GPU cannot count
// DRAM bytes. Compact-layout phases replace the 424 constant.
uint64_t b3MetalAnalyticSolverBytes( int contactCount, int substepCount );

// Expected dispatch count for the colored solve command buffer under the
// current schedule: fixed stages plus 13 dispatches per active color per
// substep group (substep x color x constraint type plus barriers). Later
// phases replace this formula with measured per-encode counts; the
// MetalTimelineTest locks the formula so that change is conscious.
int b3MetalExpectedSolveDispatchCount( int activeColorCount );

#ifdef __cplusplus
}
#endif
