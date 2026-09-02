// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "box3d/base.h"

#include "body.h"
#include "math_internal.h"
#include "metal_backend.h"

#include <stdio.h>
#include <stdlib.h>

static void IntegrateCPU( b3BodyState* states, const b3BodySim* sims, int count, float h, b3Vec3 gravity,
						  float maxLinearSpeed, float maxAngularSpeed )
{
	float maxLinearSpeedSquared = maxLinearSpeed * maxLinearSpeed;
	float maxAngularSpeedSquared = maxAngularSpeed * maxAngularSpeed;
	for ( int i = 0; i < count; ++i )
	{
		b3BodyState* state = states + i;
		const b3BodySim* sim = sims + i;
		b3Vec3 v = state->linearVelocity;
		b3Vec3 w = state->angularVelocity;
		float linearDamping = 1.0f / ( 1.0f + h * sim->linearDamping );
		float angularDamping = 1.0f / ( 1.0f + h * sim->angularDamping );
		float gravityScale = sim->invMass > 0.0f ? sim->gravityScale : 0.0f;
		v = b3MulAdd( b3Blend2( h * sim->invMass, sim->force, h * gravityScale, gravity ), linearDamping, v );
		w = b3MulAdd( b3MulSV( h, b3MulMV( sim->invInertiaWorld, sim->torque ) ), angularDamping, w );

		b3Quat q = b3MulQuat( state->deltaRotation, sim->transform.q );
		b3Matrix3 inertiaLocal = b3InvertMatrix( sim->invInertiaLocal );
		b3Vec3 omega1 = b3InvRotateVector( q, w );
		float i00 = inertiaLocal.cx.x, i01 = inertiaLocal.cy.x, i02 = inertiaLocal.cz.x;
		float i11 = inertiaLocal.cy.y, i12 = inertiaLocal.cz.y, i22 = inertiaLocal.cz.z;
		float w1 = omega1.x, w2 = omega1.y, w3 = omega1.z;
		float Iw1 = i00 * w1 + i01 * w2 + i02 * w3;
		float Iw2 = i01 * w1 + i11 * w2 + i12 * w3;
		float Iw3 = i02 * w1 + i12 * w2 + i22 * w3;
		b3Vec3 residual = { h * ( w2 * Iw3 - w3 * Iw2 ), h * ( w3 * Iw1 - w1 * Iw3 ),
							h * ( w1 * Iw2 - w2 * Iw1 ) };
		b3Matrix3 J = {
			{ i00 + h * ( w2 * i02 - w3 * i01 ), i01 + h * ( w3 * i00 - w1 * i02 - Iw3 ),
			  i02 + h * ( w1 * i01 - w2 * i00 + Iw2 ) },
			{ i01 + h * ( w2 * i12 - w3 * i11 + Iw3 ), i11 + h * ( w3 * i01 - w1 * i12 ),
			  i12 + h * ( w1 * i11 - w2 * i01 - Iw1 ) },
			{ i02 + h * ( w2 * i22 - w3 * i12 - Iw2 ), i12 + h * ( w3 * i02 - w1 * i22 + Iw1 ),
			  i22 + h * ( w1 * i12 - w2 * i02 ) },
		};
		w = b3RotateVector( q, b3Sub( omega1, b3Solve3( J, residual ) ) );

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

static void Initialize( b3BodyState* states, b3BodySim* sims, int count )
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
		float ix = 0.2f + 0.01f * (float)( i % 97 );
		float iy = 0.3f + 0.01f * (float)( i % 83 );
		float iz = 0.4f + 0.01f * (float)( i % 71 );
		sims[i].transform.q = b3Quat_identity;
		sims[i].force = (b3Vec3){ 0.3f * x, -0.2f * y, 0.1f * z };
		sims[i].torque = (b3Vec3){ 0.05f * y, 0.04f * z, -0.03f * x };
		sims[i].invMass = 0.25f + 0.01f * (float)( i % 101 );
		sims[i].invInertiaLocal = (b3Matrix3){ { ix, 0.0f, 0.0f }, { 0.0f, iy, 0.0f }, { 0.0f, 0.0f, iz } };
		sims[i].invInertiaWorld = sims[i].invInertiaLocal;
		sims[i].linearDamping = 0.1f + 0.001f * (float)( i % 29 );
		sims[i].angularDamping = 0.2f + 0.001f * (float)( i % 31 );
		sims[i].gravityScale = 1.0f;
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
	const int subStepCount = 4;
	printf( "# device=%s operation=fused_velocity_position substeps=%d timing=end-to-end_gpu_including_pack_copies_and_wait\n",
		deviceName, subStepCount );
	printf( "bodies,repeats,cpu_ms,gpu_total_ms,gpu_kernel_ms,speedup\n" );

	const int counts[] = { 32, 128, 512, 2048, 8192, 32768, 131072, 524288 };
	for ( int testIndex = 0; testIndex < (int)( sizeof( counts ) / sizeof( counts[0] ) ); ++testIndex )
	{
		int count = counts[testIndex];
		int repeats = 4000000 / count;
		repeats = repeats < 6 ? 6 : repeats;
		repeats = repeats > 1500 ? 1500 : repeats;
		b3BodyState* cpu = calloc( (size_t)count, sizeof( b3BodyState ) );
		b3BodyState* gpu = calloc( (size_t)count, sizeof( b3BodyState ) );
		b3BodySim* sims = calloc( (size_t)count, sizeof( b3BodySim ) );
		if ( cpu == NULL || gpu == NULL || sims == NULL )
		{
			return 2;
		}
		Initialize( cpu, sims, count );
		Initialize( gpu, sims, count );
		b3MetalDispatchStats stats = { 0 };
		for ( int i = 0; i < 3; ++i )
		{
			if ( b3MetalIntegrateUnconstrainedSubsteps( context, gpu, sims, count, subStepCount, 1.0f / 240.0f,
					(b3Vec3){ 0.0f, -10.0f, 0.0f }, 80.0f, 47.12389f, 0.0f, NULL, &stats ) == false )
			{
				return 3;
			}
		}

		uint64_t ticks = b3GetTicks();
		for ( int i = 0; i < repeats; ++i )
		{
			for ( int subStepIndex = 0; subStepIndex < subStepCount; ++subStepIndex )
			{
				IntegrateCPU( cpu, sims, count, 1.0f / 240.0f, (b3Vec3){ 0.0f, -10.0f, 0.0f }, 80.0f, 47.12389f );
			}
		}
		double cpuMs = b3GetMilliseconds( ticks ) / repeats;
		double kernelMs = 0.0;
		ticks = b3GetTicks();
		for ( int i = 0; i < repeats; ++i )
		{
			if ( b3MetalIntegrateUnconstrainedSubsteps( context, gpu, sims, count, subStepCount, 1.0f / 240.0f,
					(b3Vec3){ 0.0f, -10.0f, 0.0f }, 80.0f, 47.12389f, 0.0f, NULL, &stats ) == false )
			{
				return 3;
			}
			kernelMs += stats.gpuMilliseconds;
		}
		double gpuMs = b3GetMilliseconds( ticks ) / repeats;
		printf( "%d,%d,%.6f,%.6f,%.6f,%.3f\n", count, repeats, cpuMs, gpuMs, kernelMs / repeats, cpuMs / gpuMs );
		free( sims );
		free( gpu );
		free( cpu );
	}
	b3MetalDestroyContext( context );
	return 0;
}
