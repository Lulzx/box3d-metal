// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>

static b3WorldId CreateConvexStackWorld( int bodyCount, int workerCount, bool enableMetal )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.dynamicBodyCount = bodyCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	if ( enableMetal && b3World_EnableMetal( worldId, 1 ) == false )
	{
		b3DestroyWorld( worldId );
		return b3_nullWorldId;
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.6f;
	shapeDef.baseMaterial.restitution = 0.0f;
	shapeDef.baseMaterial.rollingResistance = 0.05f;
	b3BoxHull groundHull = b3MakeBoxHull( (float)bodyCount, 1.0f, 4.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId ground = b3CreateBody( worldId, &groundDef );
	b3CreateHullShape( ground, &shapeDef, &groundHull.base );

	b3BoxHull boxHull = b3MakeBoxHull( 0.4f, 0.5f, 0.4f );
	const int layerCount = 8;
	int columnCount = ( bodyCount + layerCount - 1 ) / layerCount;
	for ( int i = 0; i < bodyCount; ++i )
	{
		int column = i / layerCount;
		int layer = i % layerCount;
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ (float)column - 0.5f * (float)columnCount, 0.49f + (float)layer, 0.0f };
		b3BodyId body = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( body, &shapeDef, &boxHull.base );
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
	printf( "# operation=whole_world_convex_contacts friction=0.6 rolling=0.05 substeps=4 workers=%d timing=wall_clock_step\n",
		workerCount );
	printf( "bodies,repeats,cpu_ms,gpu_ms,metal_kernel_ms,speedup,metal_substep_dispatches\n" );
	const int counts[] = { 512, 2048, 8192, 32768, 131072, 262144 };
	for ( int testIndex = 0; testIndex < (int)( sizeof( counts ) / sizeof( counts[0] ) ); ++testIndex )
	{
		int bodyCount = counts[testIndex];
		int repeats = bodyCount <= 2048 ? 40 : bodyCount <= 8192 ? 20 : bodyCount <= 32768 ? 8 : bodyCount <= 131072 ? 4 : 2;
		b3WorldId cpuWorld = CreateConvexStackWorld( bodyCount, workerCount, false );
		double cpuMs = TimeWorld( cpuWorld, 8, repeats );
		b3DestroyWorld( cpuWorld );

		b3WorldId gpuWorld = CreateConvexStackWorld( bodyCount, workerCount, true );
		if ( B3_IS_NULL( gpuWorld ) )
		{
			fprintf( stderr, "Metal initialization failed\n" );
			return 1;
		}
		double gpuMs = TimeWorld( gpuWorld, 8, repeats );
		b3MetalProfile metal = b3World_GetMetalProfile( gpuWorld );
		printf( "%d,%d,%.6f,%.6f,%.6f,%.3f,%llu\n", bodyCount, repeats, cpuMs, gpuMs,
			metal.lastContactGpuMilliseconds, cpuMs / gpuMs,
			(unsigned long long)metal.contactDispatchCount );
		b3DestroyWorld( gpuWorld );
	}
	return 0;
}
