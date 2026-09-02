// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/base.h"

#include "body.h"
#include "math_internal.h"
#include "metal_backend.h"

#include <stdio.h>
#include <stdlib.h>

static void IntegrateCPU( b3BodyState* states, int count, float h, float maxLinearSpeed, float maxAngularSpeed )
{
	float maxLinearSpeedSquared = maxLinearSpeed * maxLinearSpeed;
	float maxAngularSpeedSquared = maxAngularSpeed * maxAngularSpeed;
	for ( int i = 0; i < count; ++i )
	{
		b3BodyState* state = states + i;
		b3Vec3 v = state->linearVelocity;
		b3Vec3 w = state->angularVelocity;
		if ( b3Dot( v, v ) > maxLinearSpeedSquared )
		{
			v = b3MulSV( maxLinearSpeed / b3Length( v ), v );
		}
		if ( b3Dot( w, w ) > maxAngularSpeedSquared )
		{
			w = b3MulSV( maxAngularSpeed / b3Length( w ), w );
		}
		state->linearVelocity = v;
		state->angularVelocity = w;
		state->deltaPosition = b3MulAdd( state->deltaPosition, h, v );
		state->deltaRotation = b3IntegrateRotation( state->deltaRotation, b3MulSV( h, w ) );
	}
}

static void InitializeStates( b3BodyState* states, int count )
{
	for ( int i = 0; i < count; ++i )
	{
		float x = (float)( ( i * 37 ) % 211 ) - 105.0f;
		float y = (float)( ( i * 73 ) % 193 ) - 96.0f;
		float z = (float)( ( i * 19 ) % 179 ) - 89.0f;
		states[i] = (b3BodyState){
			.linearVelocity = { x, y, z },
			.angularVelocity = { 0.31f * y, -0.27f * z, 0.19f * x },
			.deltaRotation = b3Quat_identity,
		};
	}
}

int main( void )
{
	b3MetalContext* context = NULL;
	char error[1024] = { 0 };
	if ( b3MetalCreateContext( &context, error, sizeof( error ) ) == false )
	{
		fprintf( stderr, "Metal initialization failed: %s\n", error );
		return 1;
	}

	char deviceName[256];
	b3MetalGetDeviceName( context, deviceName, sizeof( deviceName ) );
	printf( "# device=%s state_bytes=%zu timing=end-to-end_gpu_including_shared_memory_copies_and_wait\n", deviceName,
			sizeof( b3BodyState ) );
	printf( "bodies,repeats,cpu_ms,gpu_total_ms,gpu_kernel_ms,speedup\n" );

	const int counts[] = { 32, 128, 512, 2048, 8192, 32768, 131072, 524288 };
	const int countCount = (int)( sizeof( counts ) / sizeof( counts[0] ) );
	for ( int testIndex = 0; testIndex < countCount; ++testIndex )
	{
		int count = counts[testIndex];
		int repeats = 8000000 / count;
		repeats = repeats < 8 ? 8 : repeats;
		repeats = repeats > 2000 ? 2000 : repeats;

		b3BodyState* cpu = malloc( (size_t)count * sizeof( b3BodyState ) );
		b3BodyState* gpu = malloc( (size_t)count * sizeof( b3BodyState ) );
		if ( cpu == NULL || gpu == NULL )
		{
			fprintf( stderr, "allocation failed for %d bodies\n", count );
			return 2;
		}
		InitializeStates( cpu, count );
		InitializeStates( gpu, count );

		b3MetalDispatchStats stats = { 0 };
		for ( int i = 0; i < 3; ++i )
		{
			if ( b3MetalIntegratePositions( context, gpu, count, 1.0f / 240.0f, 80.0f, 47.12389f, &stats ) == false )
			{
				fprintf( stderr, "Metal dispatch failed for %d bodies\n", count );
				return 3;
			}
		}

		uint64_t ticks = b3GetTicks();
		for ( int i = 0; i < repeats; ++i )
		{
			IntegrateCPU( cpu, count, 1.0f / 240.0f, 80.0f, 47.12389f );
		}
		double cpuMs = b3GetMilliseconds( ticks ) / repeats;

		double kernelMs = 0.0;
		ticks = b3GetTicks();
		for ( int i = 0; i < repeats; ++i )
		{
			if ( b3MetalIntegratePositions( context, gpu, count, 1.0f / 240.0f, 80.0f, 47.12389f, &stats ) == false )
			{
				return 3;
			}
			kernelMs += stats.gpuMilliseconds;
		}
		double gpuMs = b3GetMilliseconds( ticks ) / repeats;
		kernelMs /= repeats;

		printf( "%d,%d,%.6f,%.6f,%.6f,%.3f\n", count, repeats, cpuMs, gpuMs, kernelMs, cpuMs / gpuMs );
		free( gpu );
		free( cpu );
	}

	b3MetalDestroyContext( context );
	return 0;
}
