// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include <stdio.h>

int main( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	b3WorldId world = b3CreateWorld( &worldDef );
	if ( b3World_EnableMetal( world, 1 ) == false )
	{
		fprintf( stderr, "Metal is unavailable; no GPU demo was run.\n" );
		b3DestroyWorld( world );
		return 1;
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.6f;
	shapeDef.baseMaterial.restitution = 0.1f;
	shapeDef.baseMaterial.rollingResistance = 0.05f;
	b3BoxHull groundHull = b3MakeBoxHull( 20.0f, 1.0f, 4.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId ground = b3CreateBody( world, &groundDef );
	b3CreateHullShape( ground, &shapeDef, &groundHull.base );

	b3BoxHull boxHull = b3MakeBoxHull( 0.4f, 0.5f, 0.4f );
	for ( int i = 0; i < 256; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -7.5f + (float)( i % 16 ), 0.5f + (float)( i / 16 ), 0.0f };
		b3BodyId body = b3CreateBody( world, &bodyDef );
		b3CreateHullShape( body, &shapeDef, &boxHull.base );
	}

	uint64_t ticks = b3GetTicks();
	for ( int step = 0; step < 120; ++step )
	{
		b3World_Step( world, 1.0f / 60.0f, 4 );
	}
	double elapsedMs = b3GetMilliseconds( ticks );
	b3MetalProfile profile = b3World_GetMetalProfile( world );
	printf( "device: %s\n", profile.deviceName );
	printf( "simulated: 256 bodies, 120 steps, 4 substeps in %.3f ms\n", elapsedMs );
	printf( "Metal convex-contact substeps: %llu (fallbacks: %llu)\n",
		(unsigned long long)profile.contactDispatchCount,
		(unsigned long long)profile.contactFallbackCount );
	printf( "last Metal contact command buffer: %.3f ms\n", profile.lastContactGpuMilliseconds );

	bool usedContactGpu = profile.contactDispatchCount > 0;
	b3DestroyWorld( world );
	return usedContactGpu ? 0 : 2;
}
