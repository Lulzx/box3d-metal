// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static b3WorldId CreateResidentContactWorld( int contactCount, int workerCount, bool enableMetal, bool useBoxes )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.staticBodyCount = contactCount;
	worldDef.capacity.dynamicBodyCount = contactCount;
	worldDef.capacity.staticShapeCount = contactCount;
	worldDef.capacity.dynamicShapeCount = contactCount;
	worldDef.capacity.contactCount = contactCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	if ( enableMetal && ( b3World_EnableMetal( worldId, 1 ) == false ||
		 b3World_SetMetalFinalization( worldId, true ) == false || b3World_SetMetalBroadPhase( worldId, true ) == false ) )
	{
		b3DestroyWorld( worldId );
		return b3_nullWorldId;
	}
	b3World_SetContactRecycleDistance( worldId, 0.0f );

	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.6f;
	shapeDef.baseMaterial.rollingResistance = 0.05f;
	for ( int i = 0; i < contactCount; ++i )
	{
		b3BodyDef staticDef = b3DefaultBodyDef();
		staticDef.position = (b3Pos){ 0.0, 0.0, 3.0 * (double)i };
		b3BodyId staticBody = b3CreateBody( worldId, &staticDef );
		if ( useBoxes )
			b3CreateHullShape( staticBody, &shapeDef, &box.base );
		else
			b3CreateSphereShape( staticBody, &shapeDef, &sphere );

		b3BodyDef dynamicDef = b3DefaultBodyDef();
		dynamicDef.type = b3_dynamicBody;
		dynamicDef.enableSleep = false;
		dynamicDef.position = useBoxes ? (b3Pos){ 0.0, 0.99, 3.0 * (double)i }
								   : (b3Pos){ 0.99, 0.0, 3.0 * (double)i };
		b3BodyId dynamicBody = b3CreateBody( worldId, &dynamicDef );
		if ( useBoxes )
			b3CreateHullShape( dynamicBody, &shapeDef, &box.base );
		else
			b3CreateSphereShape( dynamicBody, &shapeDef, &sphere );
	}
	return worldId;
}

static double TimeWorld( b3WorldId worldId, int warmupCount, int repeatCount )
{
	for ( int i = 0; i < warmupCount; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}
	uint64_t ticks = b3GetTicks();
	for ( int i = 0; i < repeatCount; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}
	return b3GetMilliseconds( ticks ) / repeatCount;
}

int main( void )
{
	const int workerCount = 8;
	const char* countText = getenv( "BOX3D_METAL_RESIDENT_CONTACT_COUNT" );
	const char* repeatText = getenv( "BOX3D_METAL_RESIDENT_CONTACT_REPEATS" );
	const char* shapeText = getenv( "BOX3D_METAL_RESIDENT_CONTACT_SHAPE" );
	const char* coldText = getenv( "BOX3D_METAL_RESIDENT_CONTACT_COLD_PAIR" );
	bool useBoxes = shapeText != NULL && strcmp( shapeText, "box" ) == 0;
	bool coldPair = coldText != NULL && atoi( coldText ) != 0;
	int requestedCount = countText != NULL ? atoi( countText ) : 0;
	int requestedRepeats = repeatText != NULL ? atoi( repeatText ) : 0;
	const int counts[] = { 512, 2048, 8192, 32768, 65536 };
	int testCount = requestedCount > 0 ? 1 : (int)( sizeof( counts ) / sizeof( counts[0] ) );

	printf( "# operation=whole_world_resident_%s_contacts phase=%s substeps=4 workers=%d timing=wall_clock_step\n",
		useBoxes ? "box" : "sphere", coldPair ? "cold_pair" : "steady", workerCount );
	printf( "contacts,repeats,cpu_ms,gpu_ms,speedup,prepare_dispatches,device_prepare_refreshes,collision_bypasses,cpu_collision_"
			"contacts,last_collision_exceptions,manifold_exception_bytes,input_packs,input_reuses,last_input_bytes,coverage_"
			"bypasses,state_walk_bypasses,last_state_clear_bytes,hit_clear_bypasses,last_hit_clear_bytes,awake_island_clear_"
			"bypasses,last_awake_island_clear_bytes,manifold_syncs,body_walk_bypasses,shape_applies,state_syncs,last_state_bytes,"
			"sim_syncs,last_sim_count,shape_syncs,"
			"schedule_packs,schedule_reuses,store_bypasses,event_syncs,public_syncs,index_bytes,prior_stream_bytes,impulse_bytes,"
			"prior_impulse_bytes,resident_pair_moves,enlarged_shape_traversal_bypasses,last_pair_moves,last_pair_candidates,"
			"last_pair_move_upload_bytes,pair_cpu_candidate_traversal_bypasses,last_pair_cpu_filter_moves,"
			"last_pair_cpu_filter_candidates,last_pair_direct_create_candidates,pair_contact_seed_dispatches,"
			"pair_record_traversal_bypasses,last_pair_contact_seed_count,last_pair_contact_seed_bytes\n" );
	for ( int testIndex = 0; testIndex < testCount; ++testIndex )
	{
		int contactCount = requestedCount > 0 ? requestedCount : counts[testIndex];
		int repeats = coldPair				  ? 1
					  : requestedRepeats > 0	  ? requestedRepeats
					  : contactCount <= 2048  ? 40
					  : contactCount <= 8192  ? 20
					  : contactCount <= 32768 ? 8
											  : 4;
		b3WorldId cpuWorld = CreateResidentContactWorld( contactCount, workerCount, false, useBoxes );
		double cpuMs = TimeWorld( cpuWorld, coldPair ? 0 : 8, repeats );
		b3DestroyWorld( cpuWorld );

		b3WorldId gpuWorld = CreateResidentContactWorld( contactCount, workerCount, true, useBoxes );
		if ( B3_IS_NULL( gpuWorld ) )
		{
			fprintf( stderr, "Metal initialization failed\n" );
			return 1;
		}
		double gpuMs = TimeWorld( gpuWorld, coldPair ? 0 : 8, repeats );
		b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
		uint64_t priorBytes = profile.lastContactPrepareIndexBytes / sizeof( uint32_t ) * 144;
		uint64_t priorImpulseBytes = (uint64_t)profile.lastResidentConvexConstraintCount * 1696;
		printf( "%d,%d,%.6f,%.6f,%.3f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%"
				"llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,%llu,%llu,%d,%d,%d,%llu,%llu,%d,%llu\n",
				contactCount, repeats, cpuMs, gpuMs, cpuMs / gpuMs, (unsigned long long)profile.contactPrepareDispatchCount,
				(unsigned long long)profile.contactPrepareDeviceRefreshCount,
				(unsigned long long)profile.contactCollisionBypassCount, (unsigned long long)profile.contactCollisionCpuCount,
				(unsigned long long)profile.lastContactCollisionExceptionCount,
				(unsigned long long)profile.lastNarrowPhaseResultBytes, (unsigned long long)profile.contactInputPackCount,
				(unsigned long long)profile.contactInputReuseCount, (unsigned long long)profile.lastContactInputBytes,
				(unsigned long long)profile.contactCoverageBypassCount,
				(unsigned long long)profile.contactStateTraversalBypassCount,
				(unsigned long long)profile.lastContactStateBitSetBytes,
				(unsigned long long)profile.contactHitEventBitSetClearBypassCount,
				(unsigned long long)profile.lastContactHitEventBitSetBytes,
				(unsigned long long)profile.awakeIslandBitSetClearBypassCount,
				(unsigned long long)profile.lastAwakeIslandBitSetBytes,
				(unsigned long long)profile.contactManifoldSyncCount,
				(unsigned long long)profile.finalizationBodyTraversalBypassCount,
				(unsigned long long)profile.shapeResultApplyCount,
				(unsigned long long)profile.bodyStateSyncCount,
				(unsigned long long)profile.lastBodyStateReadbackBytes,
				(unsigned long long)profile.bodySimSyncCount,
				(unsigned long long)profile.lastBodySimSyncCount,
				(unsigned long long)profile.shapeBoundsSyncCount,
				(unsigned long long)profile.contactSchedulePackCount, (unsigned long long)profile.contactScheduleReuseCount,
				(unsigned long long)profile.contactImpulseStoreBypassCount,
				(unsigned long long)profile.contactImpulseEventSyncCount, (unsigned long long)profile.contactImpulseSyncCount,
				(unsigned long long)profile.lastContactPrepareIndexBytes, (unsigned long long)priorBytes,
				(unsigned long long)profile.lastContactImpulseResultBytes, (unsigned long long)priorImpulseBytes,
				(unsigned long long)profile.residentPairMoveDispatchCount,
				(unsigned long long)profile.enlargedShapeTraversalBypassCount,
				profile.lastPairMoveCount, profile.lastPairCandidateCount,
				(unsigned long long)profile.lastPairMoveUploadBytes,
				(unsigned long long)profile.pairCpuCandidateTraversalBypassCount,
				profile.lastPairCpuFilterMoveCount, profile.lastPairCpuFilterCandidateCount,
				profile.lastPairDirectCreateCount,
				(unsigned long long)profile.pairContactSeedDispatchCount,
				(unsigned long long)profile.pairRecordTraversalBypassCount,
				profile.lastPairContactSeedCount, (unsigned long long)profile.lastPairContactSeedBytes );
		b3DestroyWorld( gpuWorld );
	}
	return 0;
}
