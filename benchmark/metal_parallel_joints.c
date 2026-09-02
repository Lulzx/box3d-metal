// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>

static b3WorldId CreateParallelJointWorld( int bodyCount, int workerCount, bool enableMetal )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -1.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	worldDef.workerCount = (uint32_t)workerCount;
	worldDef.capacity.dynamicBodyCount = bodyCount;
	b3WorldId world = b3CreateWorld( &worldDef );
	if ( enableMetal && b3World_EnableMetal( world, 1 ) == false )
	{
		b3DestroyWorld( world );
		return b3_nullWorldId;
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3BoxHull hull = b3MakeBoxHull( 0.18f, 0.25f, 0.2f );
	for ( int pair = 0; pair < bodyCount / 2; ++pair )
	{
		float x = 2.5f * (float)( pair % 512 );
		float z = 2.5f * (float)( pair / 512 );
		b3BodyDef bodyDefA = b3DefaultBodyDef();
		bodyDefA.type = b3_dynamicBody;
		bodyDefA.enableSleep = false;
		bodyDefA.position = (b3Pos){ x, 0.0f, z };
		bodyDefA.angularVelocity = (b3Vec3){ 0.02f, -0.03f, 0.01f };
		b3BodyDef bodyDefB = bodyDefA;
		bodyDefB.position.x += 1.2f;
		bodyDefB.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 1.0f, 0.4f, 0.2f } ),
			0.08f + 0.00001f * (float)( pair % 1000 ) );
		bodyDefB.angularVelocity = (b3Vec3){ -0.05f, 0.04f, -0.02f };
		b3BodyId bodyA = b3CreateBody( world, &bodyDefA );
		b3BodyId bodyB = b3CreateBody( world, &bodyDefB );
		b3CreateHullShape( bodyA, &shapeDef, &hull.base );
		b3CreateHullShape( bodyB, &shapeDef, &hull.base );

		b3ParallelJointDef jointDef = b3DefaultParallelJointDef();
		jointDef.base.bodyIdA = bodyA;
		jointDef.base.bodyIdB = bodyB;
		jointDef.hertz = 3.0f;
		jointDef.dampingRatio = 0.7f;
		jointDef.maxTorque = 30.0f;
		b3CreateParallelJoint( world, &jointDef );
	}
	return world;
}

static double TimeWorld( b3WorldId world, int warmups, int repeats )
{
	for ( int i = 0; i < warmups; ++i ) b3World_Step( world, 1.0f / 60.0f, 4 );
	uint64_t ticks = b3GetTicks();
	for ( int i = 0; i < repeats; ++i ) b3World_Step( world, 1.0f / 60.0f, 4 );
	return b3GetMilliseconds( ticks ) / repeats;
}

int main( int argc, char** argv )
{
	const int workerCount = 8;
	printf( "# operation=whole_world_parallel_joints substeps=4 workers=%d timing=wall_clock_step\n", workerCount );
	printf( "bodies,joints,repeats,cpu_ms,gpu_ms,metal_kernel_ms,speedup,metal_substep_dispatches\n" );
	fflush( stdout );
	const int counts[] = { 512, 2048, 8192, 32768, 131072, 524288, 1048576 };
	for ( int testIndex = 0; testIndex < (int)( sizeof( counts ) / sizeof( counts[0] ) ); ++testIndex )
	{
		int bodyCount = counts[testIndex];
		if ( argc > 1 && bodyCount != atoi( argv[1] ) ) continue;
		int repeats = bodyCount <= 2048 ? 40 : bodyCount <= 8192 ? 20 : bodyCount <= 32768 ? 8 : bodyCount <= 131072 ? 4 :
			bodyCount <= 524288 ? 2 : 1;
		b3WorldId cpu = CreateParallelJointWorld( bodyCount, workerCount, false );
		double cpuMs = TimeWorld( cpu, 8, repeats );
		b3DestroyWorld( cpu );
		b3WorldId gpu = CreateParallelJointWorld( bodyCount, workerCount, true );
		if ( B3_IS_NULL( gpu ) )
		{
			fprintf( stderr, "Metal initialization failed\n" );
			return 1;
		}
		double gpuMs = TimeWorld( gpu, 8, repeats );
		b3MetalProfile metal = b3World_GetMetalProfile( gpu );
		printf( "%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%llu\n", bodyCount, bodyCount / 2, repeats, cpuMs, gpuMs,
			metal.lastJointGpuMilliseconds, cpuMs / gpuMs, (unsigned long long)metal.jointDispatchCount );
		fflush( stdout );
		b3DestroyWorld( gpu );
	}
	return 0;
}
