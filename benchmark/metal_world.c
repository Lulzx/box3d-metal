// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>

static int ReadPositiveEnvironment( const char* name )
{
	const char* text = getenv( name );
	if ( text == NULL ) return 0;
	char* end = NULL;
	long value = strtol( text, &end, 10 );
	if ( end == text || *end != '\0' || value < 1 || value > INT32_MAX )
	{
		fprintf( stderr, "%s must be an integer in [1, %d]\n", name, INT32_MAX );
		exit( 2 );
	}
	return (int)value;
}

static b3WorldId CreateBallisticWorld( int bodyCount, int workerCount, bool createShapes )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.dynamicBodyCount = bodyCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	shapeDef.invokeContactCreation = false;
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ (float)( i % 256 ), (float)( ( i / 256 ) % 256 ), (float)( i / 65536 ) };
		bodyDef.linearVelocity = (b3Vec3){ 0.01f * (float)( i % 97 ), 2.0f, -0.01f * (float)( i % 89 ) };
		bodyDef.angularVelocity = (b3Vec3){ 0.1f, 0.2f, 0.3f };
		b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
		if ( createShapes )
		{
			b3Sphere sphere = { .center = { 0.1f, -0.05f, 0.08f }, .radius = 0.25f };
			b3CreateSphereShape( bodyId, &shapeDef, &sphere );
		}
	}
	return worldId;
}

static double TimeWorld( b3WorldId worldId, int repeats )
{
	for ( int i = 0; i < 5; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}
	uint64_t ticks = b3GetTicks();
	for ( int i = 0; i < repeats; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}
	return b3GetMilliseconds( ticks ) / repeats;
}

int main( void )
{
	const int workerCount = 8;
	bool enableFinalization = getenv( "BOX3D_METAL_FINALIZATION" ) != NULL;
	bool enableBroadPhase = getenv( "BOX3D_METAL_BROAD_PHASE" ) != NULL;
	bool createShapes = getenv( "BOX3D_METAL_SHAPES" ) != NULL;
	int selectedBodyCount = ReadPositiveEnvironment( "BOX3D_METAL_WORLD_COUNT" );
	int selectedRepeats = ReadPositiveEnvironment( "BOX3D_METAL_WORLD_REPEATS" );
	printf( "# operation=whole_world_unconstrained substeps=4 workers=%d timing=wall_clock_step metal_finalization=%s "
		"metal_broad_phase=%s shapes=%s\n", workerCount, enableFinalization ? "on" : "off",
		enableBroadPhase ? "on" : "off", createShapes ? "sphere_per_body" : "none" );
	printf( "bodies,repeats,cpu_ms,gpu_ms,metal_kernel_ms,finalization_readback_bytes,finalization_readback_bypasses,"
		"finalization_shape_traversal_bypasses,move_event_dispatches,move_event_syncs,last_move_event_readback_bytes,"
		"transform_device_refreshes,pair_kernel_ms,pair_dispatches,"
		"body_state_uploads,body_state_reuses,last_body_state_upload_bytes,body_state_revision_checks,body_state_syncs,last_body_state_readback_bytes,"
		"body_property_uploads,body_property_reuses,last_body_property_upload_bytes,pair_fallbacks,speedup\n" );
	const int counts[] = { 512, 2048, 8192, 32768, 131072, 524288 };
	int testCount = selectedBodyCount > 0 ? 1 : (int)( sizeof( counts ) / sizeof( counts[0] ) );
	for ( int testIndex = 0; testIndex < testCount; ++testIndex )
	{
		int bodyCount = selectedBodyCount > 0 ? selectedBodyCount : counts[testIndex];
		int repeats = selectedRepeats > 0 ? selectedRepeats :
			bodyCount <= 8192 ? 80 : bodyCount <= 32768 ? 40 : bodyCount <= 131072 ? 16 : 6;

		b3WorldId cpuWorld = CreateBallisticWorld( bodyCount, workerCount, createShapes );
		double cpuMs = TimeWorld( cpuWorld, repeats );
		b3DestroyWorld( cpuWorld );

		b3WorldId gpuWorld = CreateBallisticWorld( bodyCount, workerCount, createShapes );
		if ( b3World_EnableMetal( gpuWorld, 1 ) == false )
		{
			fprintf( stderr, "Metal initialization failed\n" );
			return 1;
		}
		if ( enableFinalization && b3World_SetMetalFinalization( gpuWorld, true ) == false )
		{
			fprintf( stderr, "Metal finalization enable failed\n" );
			return 1;
		}
		if ( enableBroadPhase && b3World_SetMetalBroadPhase( gpuWorld, true ) == false )
		{
			fprintf( stderr, "Metal broad phase enable failed\n" );
			return 1;
		}
		double gpuMs = TimeWorld( gpuWorld, repeats );
		b3MetalProfile metal = b3World_GetMetalProfile( gpuWorld );
		printf( "%d,%d,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.3f\n",
			bodyCount,
			repeats, cpuMs, gpuMs,
			metal.lastUnconstrainedGpuMilliseconds, (unsigned long long)metal.lastFinalizationReadbackBytes,
			(unsigned long long)metal.finalizationReadbackBypassCount,
			(unsigned long long)metal.finalizationShapeTraversalBypassCount,
			(unsigned long long)metal.bodyMoveEventDispatchCount,
			(unsigned long long)metal.bodyMoveEventSyncCount,
			(unsigned long long)metal.lastBodyMoveEventReadbackBytes,
			(unsigned long long)metal.narrowPhaseTransformDeviceRefreshCount,
			metal.lastPairGpuMilliseconds,
			(unsigned long long)metal.pairDispatchCount, (unsigned long long)metal.bodyStateUploadCount,
			(unsigned long long)metal.bodyStateReuseCount, (unsigned long long)metal.lastBodyStateUploadBytes,
			(unsigned long long)metal.bodyStateRevisionCheckCount,
			(unsigned long long)metal.bodyStateSyncCount,
			(unsigned long long)metal.lastBodyStateReadbackBytes,
			(unsigned long long)metal.bodyPropertyUploadCount, (unsigned long long)metal.bodyPropertyReuseCount,
			(unsigned long long)metal.lastBodyPropertyUploadBytes,
			(unsigned long long)metal.pairFallbackCount, cpuMs / gpuMs );
		b3DestroyWorld( gpuWorld );
	}
	return 0;
}
