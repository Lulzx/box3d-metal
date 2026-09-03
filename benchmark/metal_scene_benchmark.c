// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

// Phase 0 scene harness: runs the realistic scenes from shared/benchmarks.h on
// CPU and Metal and reports wall time plus the new timeline profile fields.
// Today most of these scenes exercise the CPU fallback path; that fallback
// cost is the honest baseline later phases must beat. See
// docs/benchmarks/protocol.md for the quiet-host procedure.
//
// Columns:
//   scene, bodies, contacts, mode, min_ms, median_ms,
//   cmd_buffers, dispatches, barriers, encode_ms, wait_ms,
//   stage_ms[9] (mutations..events), analytic_solver_bytes, GB/s,
//   contact_fallbacks, pair_fallbacks, narrow_fallbacks

#include "benchmarks.h"

#include "box3d/box3d.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void CapacityFcn( b3Capacity* capacity );
typedef void CreateFcn( b3WorldId worldId );
typedef void StepFcn( b3WorldId worldId, int stepCount );
typedef void DestroyFcn( void );

typedef struct SceneDef
{
	const char* name;
	CapacityFcn* capacityFcn;
	CreateFcn* createFcn;
	StepFcn* stepFcn;
	DestroyFcn* destroyFcn;
} SceneDef;

static int CompareDouble( const void* a, const void* b )
{
	double x = *(const double*)a, y = *(const double*)b;
	return ( x > y ) - ( x < y );
}

static void CreateSceneWorld( const SceneDef* scene, bool enableMetal, int workerCount, b3WorldId* outWorld )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = (uint32_t)workerCount;
	if ( scene->capacityFcn != NULL )
	{
		scene->capacityFcn( &worldDef.capacity );
	}
	b3WorldId worldId = b3CreateWorld( &worldDef );
	scene->createFcn( worldId );
	if ( enableMetal )
	{
		if ( b3World_EnableMetal( worldId, 1 ) == false )
		{
			b3DestroyWorld( worldId );
			*outWorld = b3_nullWorldId;
			return;
		}
		if ( getenv( "BOX3D_METAL_FINALIZATION" ) != NULL )
		{
			b3World_SetMetalFinalization( worldId, true );
		}
		if ( getenv( "BOX3D_METAL_BROAD_PHASE" ) != NULL )
		{
			b3World_SetMetalBroadPhase( worldId, true );
		}
	}
	*outWorld = worldId;
}

static void StepScene( const SceneDef* scene, b3WorldId worldId, int stepCount )
{
	// Scene step functions only drive kinematic inputs (rain spawners, junkyard
	// pusher, falling spheres); the physics step always runs here.
	if ( scene->stepFcn != NULL )
	{
		scene->stepFcn( worldId, stepCount );
	}
	b3World_Step( worldId, 1.0f / 60.0f, 4 );
}

static void RunScene( const SceneDef* scene, int workerCount )
{
	static const char* kStageNames[9] = {
		"mutations", "broadPhase", "narrowPhase", "topology", "prepare", "solve", "finalize", "refit", "events"
	};
	(void)kStageNames;

	// Spike knob: settle dynamic scenes before measuring so steady resident
	// steps (reused inputs, zero exceptions) can be observed. Default 3
	// preserves historical numbers.
	int warmupSteps = 3;
	const char* warmupEnv = getenv( "BOX3D_METAL_WARMUP" );
	if ( warmupEnv != NULL )
	{
		warmupSteps = atoi( warmupEnv );
		if ( warmupSteps < 0 )
			warmupSteps = 0;
	}

	for ( int mode = 0; mode < 2; ++mode )
	{
		bool enableMetal = mode == 1;
		b3WorldId worldId;
		CreateSceneWorld( scene, enableMetal, workerCount, &worldId );
		if ( B3_IS_NULL( worldId ) )
		{
			printf( "%s,0,0,%s,SKIPPED\n", scene->name, enableMetal ? "metal" : "cpu" );
			continue;
		}

		for ( int i = 0; i < warmupSteps; ++i )
		{
			StepScene( scene, worldId, i );
		}
		double samples[5];
		b3MetalProfile lastProfile = { 0 };
		for ( int i = 0; i < 5; ++i )
		{
			uint64_t ticks = b3GetTicks();
			StepScene( scene, worldId, 3 + i );
			samples[i] = b3GetMilliseconds( ticks );
			if ( enableMetal )
			{
				lastProfile = b3World_GetMetalProfile( worldId );
			}
		}
		qsort( samples, 5, sizeof( double ), CompareDouble );
		double median = samples[2], min = samples[0];

		int contactCount = 0;
		{
			// GPU-resident colored convex contacts on the last step. CPU rows
			// and fallback routes report 0; use the scene's known size (see
			// docs/benchmarks) for cross-scene comparison.
			contactCount = lastProfile.lastResidentConvexContactCount;
		}

		double seconds = median / 1000.0;
		double gbps = seconds > 0.0 ? (double)lastProfile.lastAnalyticSolverBytes / seconds / 1e9 : 0.0;

		printf( "%s,%d,%d,%s,%.4f,%.4f,%llu,%llu,%llu,%.4f,%.4f,"
			"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%llu,%.3f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
			scene->name, contactCount, lastProfile.lastResidentConvexConstraintCount,
			enableMetal ? "metal" : "cpu", min, median,
			(unsigned long long)lastProfile.lastCommandBufferCount,
			(unsigned long long)lastProfile.lastDispatchCount,
			(unsigned long long)lastProfile.lastBarrierCount,
			lastProfile.lastEncodeCpuMs, lastProfile.lastWaitCpuMs,
			lastProfile.stageGpuMs[0], lastProfile.stageGpuMs[1], lastProfile.stageGpuMs[2],
			lastProfile.stageGpuMs[3], lastProfile.stageGpuMs[4], lastProfile.stageGpuMs[5],
			lastProfile.stageGpuMs[6], lastProfile.stageGpuMs[7], lastProfile.stageGpuMs[8],
			(unsigned long long)lastProfile.lastAnalyticSolverBytes, gbps,
			(unsigned long long)lastProfile.contactFallbackCount,
			(unsigned long long)lastProfile.pairFallbackCount,
			(unsigned long long)lastProfile.narrowPhaseFallbackCount,
			(unsigned long long)lastProfile.lastContactCollisionExceptionCount,
			(unsigned long long)lastProfile.contactCollisionBypassCount,
			(unsigned long long)lastProfile.mergedNarrowSolveAttemptCount,
			(unsigned long long)lastProfile.mergedNarrowSolveAcceptCount,
			(unsigned long long)lastProfile.mergedNarrowSolveMispredictCount );

		b3DestroyWorld( worldId );
		if ( scene->destroyFcn != NULL )
		{
			scene->destroyFcn();
		}
	}
}

int main( void )
{
	const int workerCount = 8;
	printf( "# operation=scene_cpu_vs_metal substeps=4 workers=%d timing=wall_clock_step "
		"repeats=5(min+median) metal_opts=env(BOX3D_METAL_FINALIZATION,BOX3D_METAL_BROAD_PHASE)\n",
		workerCount );
	printf( "scene,resident_contacts,resident_constraints,mode,min_ms,median_ms,cmd_buffers,dispatches,barriers,encode_ms,wait_ms,"
		"stage_mutations_ms,stage_broad_ms,stage_narrow_ms,stage_topology_ms,stage_prepare_ms,stage_solve_ms,"
		"stage_finalize_ms,stage_refit_ms,stage_events_ms,analytic_solver_bytes,gbps,"
		"contact_fallbacks,pair_fallbacks,narrow_fallbacks,exceptions,bypass,merged_attempts,merged_accepts,merged_mispredicts\n" );

	SceneDef scenes[] = {
		{ "large_pyramid", NULL, CreateLargePyramid, NULL, NULL },
		{ "many_pyramids", NULL, CreateManyPyramids, NULL, NULL },
		{ "rain", GetRainCapacity, CreateRain, StepRain, DestroyRain },
		{ "junkyard", GetJunkyardCapacity, CreateJunkyard, StepJunkyard, NULL },
		{ "convex_pile", GetConvexPileCapacity, CreateConvexPile, NULL, NULL },
		{ "joint_grid", NULL, CreateJointGrid, NULL, NULL },
		{ "large_world", GetLargeWorldCapacity, CreateLargeWorld, StepLargeWorld, NULL },
	};

	const char* only = getenv( "BOX3D_METAL_SCENE" );
	for ( size_t i = 0; i < sizeof( scenes ) / sizeof( scenes[0] ); ++i )
	{
		if ( only != NULL && only[0] != '\0' && strcmp( only, scenes[i].name ) != 0 )
		{
			continue;
		}
		RunScene( scenes + i, workerCount );
	}
	return 0;
}
