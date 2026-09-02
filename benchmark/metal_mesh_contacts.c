// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct MeshWorld
{
	b3WorldId world;
	b3MeshData* mesh;
} MeshWorld;

static MeshWorld CreateMeshWorld( int bodyCount, int workerCount, bool enableMetal )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.dynamicBodyCount = bodyCount;
	b3WorldId world = b3CreateWorld( &worldDef );
	if ( enableMetal && b3World_EnableMetal( world, 1 ) == false )
	{
		b3DestroyWorld( world );
		return (MeshWorld){ b3_nullWorldId, NULL };
	}

	// Body placement below stays within roughly 600 by 300 world units even at
	// the largest count. Avoid an unnecessarily enormous triangle whose float
	// geometry becomes ill-conditioned during mesh baking.
	float extent = 1024.0f;
	b3Vec3 vertices[4] = {
		{ -extent, 0.0f, -extent }, { extent, 0.0f, -extent }, { extent, 0.0f, extent }, { -extent, 0.0f, extent },
	};
	int32_t indices[6] = { 0, 2, 1, 0, 3, 2 };
	b3MeshDef meshDef = { 0 };
	meshDef.vertices = vertices;
	meshDef.stride = sizeof( b3Vec3 );
	meshDef.indices = indices;
	meshDef.vertexCount = 4;
	meshDef.triangleCount = 2;
	meshDef.identifyEdges = true;
	b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );
	if ( mesh == NULL )
	{
		b3DestroyWorld( world );
		return (MeshWorld){ b3_nullWorldId, NULL };
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.6f;
	shapeDef.baseMaterial.rollingResistance = 0.05f;
	b3BodyDef groundDef = b3DefaultBodyDef();
	b3BodyId ground = b3CreateBody( world, &groundDef );
	b3CreateMeshShape( ground, &shapeDef, mesh, (b3Vec3){ 1.0f, 1.0f, 1.0f } );

	b3BoxHull boxHull = b3MakeBoxHull( 0.4f, 0.5f, 0.4f );
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ 1.1f * (float)( i % 512 ), 0.49f, 1.1f * (float)( i / 512 ) };
		b3BodyId body = b3CreateBody( world, &bodyDef );
		b3CreateHullShape( body, &shapeDef, &boxHull.base );
	}
	return (MeshWorld){ world, mesh };
}

static void DestroyMeshWorld( MeshWorld data )
{
	b3DestroyWorld( data.world );
	b3DestroyMesh( data.mesh );
}

static double TimeWorld( b3WorldId world, int warmupCount, int repeatCount )
{
	for ( int i = 0; i < warmupCount; ++i ) b3World_Step( world, 1.0f / 60.0f, 4 );
	uint64_t ticks = b3GetTicks();
	for ( int i = 0; i < repeatCount; ++i ) b3World_Step( world, 1.0f / 60.0f, 4 );
	return b3GetMilliseconds( ticks ) / repeatCount;
}

int main( int argc, char** argv )
{
	const int workerCount = 8;
	printf( "# operation=whole_world_mesh_contacts friction=0.6 rolling=0.05 substeps=4 workers=%d timing=wall_clock_step\n",
		workerCount );
	printf( "bodies,repeats,cpu_ms,gpu_ms,metal_kernel_ms,speedup,metal_substep_dispatches\n" );
	fflush( stdout );
	const int counts[] = { 512, 2048, 8192, 32768, 131072 };
	for ( int testIndex = 0; testIndex < (int)( sizeof( counts ) / sizeof( counts[0] ) ); ++testIndex )
	{
		int bodyCount = counts[testIndex];
		if ( argc > 1 && bodyCount != atoi( argv[1] ) ) continue;
		int repeats = bodyCount <= 2048 ? 40 : bodyCount <= 8192 ? 20 : bodyCount <= 32768 ? 8 : 4;
		MeshWorld cpu = CreateMeshWorld( bodyCount, workerCount, false );
		if ( B3_IS_NULL( cpu.world ) )
		{
			fprintf( stderr, "CPU mesh world creation failed\n" );
			return 1;
		}
		double cpuMs = TimeWorld( cpu.world, 8, repeats );
		DestroyMeshWorld( cpu );

		MeshWorld gpu = CreateMeshWorld( bodyCount, workerCount, true );
		if ( B3_IS_NULL( gpu.world ) )
		{
			fprintf( stderr, "Metal initialization failed\n" );
			return 1;
		}
		double gpuMs = TimeWorld( gpu.world, 8, repeats );
		b3MetalProfile metal = b3World_GetMetalProfile( gpu.world );
		printf( "%d,%d,%.6f,%.6f,%.6f,%.3f,%llu\n", bodyCount, repeats, cpuMs, gpuMs,
			metal.lastContactGpuMilliseconds, cpuMs / gpuMs, (unsigned long long)metal.contactDispatchCount );
		fflush( stdout );
		DestroyMeshWorld( gpu );
	}
	return 0;
}
