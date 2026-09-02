// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>

static b3WorldId CreateBallisticWorld( int bodyCount, int workerCount )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.dynamicBodyCount = bodyCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ (float)( i % 256 ), (float)( ( i / 256 ) % 256 ), (float)( i / 65536 ) };
		bodyDef.linearVelocity = (b3Vec3){ 0.01f * (float)( i % 97 ), 2.0f, -0.01f * (float)( i % 89 ) };
		bodyDef.angularVelocity = (b3Vec3){ 0.1f, 0.2f, 0.3f };
		b3CreateBody( worldId, &bodyDef );
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
	printf( "# operation=whole_world_unconstrained substeps=4 workers=%d timing=wall_clock_step metal_finalization=%s\n",
		workerCount, enableFinalization ? "on" : "off" );
	printf( "bodies,repeats,cpu_ms,gpu_ms,metal_kernel_ms,speedup\n" );
	const int counts[] = { 512, 2048, 8192, 32768, 131072, 524288 };
	for ( int testIndex = 0; testIndex < (int)( sizeof( counts ) / sizeof( counts[0] ) ); ++testIndex )
	{
		int bodyCount = counts[testIndex];
		int repeats = bodyCount <= 8192 ? 80 : bodyCount <= 32768 ? 40 : bodyCount <= 131072 ? 16 : 6;

		b3WorldId cpuWorld = CreateBallisticWorld( bodyCount, workerCount );
		double cpuMs = TimeWorld( cpuWorld, repeats );
		b3DestroyWorld( cpuWorld );

		b3WorldId gpuWorld = CreateBallisticWorld( bodyCount, workerCount );
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
		double gpuMs = TimeWorld( gpuWorld, repeats );
		b3MetalProfile metal = b3World_GetMetalProfile( gpuWorld );
		printf( "%d,%d,%.6f,%.6f,%.6f,%.3f\n", bodyCount, repeats, cpuMs, gpuMs,
			metal.lastUnconstrainedGpuMilliseconds, cpuMs / gpuMs );
		b3DestroyWorld( gpuWorld );
	}
	return 0;
}
