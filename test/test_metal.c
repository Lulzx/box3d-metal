// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#include "body.h"
#include "broad_phase.h"
#include "contact_solver.h"
#include "manifold.h"
#include "math_internal.h"
#include "metal_backend.h"
#include "physics_world.h"
#include "shape.h"
#include "test_macros.h"

#include "box3d/box3d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t b3MetalRandomState = 0x92d68ca2u;

static float b3MetalRandomFloat( float low, float high )
{
	b3MetalRandomState = 1664525u * b3MetalRandomState + 1013904223u;
	float unit = (float)( b3MetalRandomState >> 8 ) * ( 1.0f / 16777216.0f );
	return low + ( high - low ) * unit;
}

typedef struct b3PairCandidateCapture
{
	b3MetalPairCandidate* candidates;
	int capacity;
	int count;
	int treeType;
	int queryProxyKey;
	int queryShapeIndex;
	const b3BroadPhase* broadPhase;
	const b3World* world;
} b3PairCandidateCapture;

static bool CapturePairCandidate( int proxyId, uint64_t userData, void* context )
{
	b3PairCandidateCapture* capture = context;
	int targetProxyKey = B3_PROXY_KEY( proxyId, capture->treeType );
	if ( targetProxyKey == capture->queryProxyKey )
		return true;
	b3BodyType queryType = B3_PROXY_TYPE( capture->queryProxyKey );
	bool targetMoved = b3GetBit( capture->broadPhase->movedProxies + capture->treeType, proxyId );
	if ( ( queryType == b3_dynamicBody && capture->treeType == b3_dynamicBody && targetProxyKey < capture->queryProxyKey &&
		   targetMoved ) ||
		 ( queryType != b3_dynamicBody && targetMoved ) )
	{
		return true;
	}
	int targetShapeIndex = (int)userData;
	const b3Shape* queryShape = capture->world->shapes.data + capture->queryShapeIndex;
	const b3Shape* targetShape = capture->world->shapes.data + targetShapeIndex;
	if ( queryShape->bodyId == targetShape->bodyId || queryShape->sensorIndex != B3_NULL_INDEX ||
		 targetShape->sensorIndex != B3_NULL_INDEX || b3ShouldShapesCollide( queryShape->filter, targetShape->filter ) == false )
	{
		return true;
	}
	if ( targetShape->type != b3_compoundShape &&
		 b3ContainsKey( &capture->broadPhase->pairSet, b3ShapePairKey( targetShapeIndex, capture->queryShapeIndex, 0 ) ) )
	{
		return true;
	}
	if ( capture->count >= capture->capacity )
		return false;
	capture->candidates[capture->count++] = (b3MetalPairCandidate){
		.proxyId = proxyId,
		.treeType = capture->treeType,
		.shapeIndex = (int)userData,
	};
	return true;
}

static int VerifyResidentPairTraversal( b3World* world )
{
	b3BroadPhase* broadPhase = &world->broadPhase;
	int moveCount = broadPhase->moveArray.count;
	ENSURE( moveCount > 0 );
	const b3MetalPairQueryRecord* records = NULL;
	const b3MetalPairCandidate* candidates = NULL;
	int candidateCount = 0;
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, moveCount, &records,
										   &candidates, &candidateCount, &stats ) );
	ENSURE( stats.treeUploadCount == 0 );
	int capacity = b3MaxInt( 1, candidateCount );
	b3MetalPairCandidate* reference = malloc( (size_t)capacity * sizeof( b3MetalPairCandidate ) );
	ENSURE( reference != NULL );
	b3PairCandidateCapture capture = {
		.candidates = reference,
		.capacity = capacity,
		.broadPhase = broadPhase,
		.world = world,
	};
	int totalCount = 0;
	for ( int moveIndex = 0; moveIndex < moveCount; ++moveIndex )
	{
		int proxyKey = broadPhase->moveArray.data[moveIndex];
		capture.queryProxyKey = proxyKey;
		b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
		int proxyId = B3_PROXY_ID( proxyKey );
		b3AABB aabb = b3DynamicTree_GetAABB( broadPhase->trees + proxyType, proxyId );
		ENSURE( records[moveIndex].queryShapeIndex == (int)b3DynamicTree_GetUserData( broadPhase->trees + proxyType, proxyId ) );
		capture.queryShapeIndex = records[moveIndex].queryShapeIndex;
		ENSURE( records[moveIndex].lowerX == aabb.lowerBound.x );
		ENSURE( records[moveIndex].lowerY == aabb.lowerBound.y );
		ENSURE( records[moveIndex].lowerZ == aabb.lowerBound.z );
		ENSURE( records[moveIndex].upperX == aabb.upperBound.x );
		ENSURE( records[moveIndex].upperY == aabb.upperBound.y );
		ENSURE( records[moveIndex].upperZ == aabb.upperBound.z );
		capture.count = 0;
		if ( proxyType == b3_dynamicBody )
		{
			capture.treeType = b3_kinematicBody;
			b3DynamicTree_Query( broadPhase->trees + b3_kinematicBody, aabb, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
								 &capture );
			capture.treeType = b3_staticBody;
			b3DynamicTree_Query( broadPhase->trees + b3_staticBody, aabb, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
								 &capture );
		}
		capture.treeType = b3_dynamicBody;
		b3DynamicTree_Query( broadPhase->trees + b3_dynamicBody, aabb, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
							 &capture );
		ENSURE( records[moveIndex].count == (uint32_t)capture.count );
		for ( int i = 0; i < capture.count; ++i )
		{
			const b3MetalPairCandidate* gpu = candidates + records[moveIndex].offset + i;
			ENSURE( gpu->proxyId == reference[i].proxyId );
			ENSURE( gpu->treeType == reference[i].treeType );
			ENSURE( gpu->shapeIndex == reference[i].shapeIndex );
		}
		totalCount += capture.count;
	}
	ENSURE( totalCount == candidateCount );
	free( reference );
	return 0;
}

static void b3IntegratePositionsReference( b3BodyState* states, int count, float h, float maxLinearSpeed, float maxAngularSpeed )
{
	float maxLinearSpeedSquared = maxLinearSpeed * maxLinearSpeed;
	float maxAngularSpeedSquared = maxAngularSpeed * maxAngularSpeed;
	for ( int i = 0; i < count; ++i )
	{
		b3BodyState* state = states + i;
		b3Vec3 v = state->linearVelocity;
		b3Vec3 w = state->angularVelocity;

		v.x = ( state->flags & b3_lockLinearX ) ? 0.0f : v.x;
		v.y = ( state->flags & b3_lockLinearY ) ? 0.0f : v.y;
		v.z = ( state->flags & b3_lockLinearZ ) ? 0.0f : v.z;
		w.x = ( state->flags & b3_lockAngularX ) ? 0.0f : w.x;
		w.y = ( state->flags & b3_lockAngularY ) ? 0.0f : w.y;
		w.z = ( state->flags & b3_lockAngularZ ) ? 0.0f : w.z;

		if ( b3Dot( v, v ) > maxLinearSpeedSquared )
		{
			v = b3MulSV( maxLinearSpeed / b3Length( v ), v );
			state->flags |= b3_isSpeedCapped;
		}
		if ( b3Dot( w, w ) > maxAngularSpeedSquared && ( state->flags & b3_allowFastRotation ) == 0 )
		{
			w = b3MulSV( maxAngularSpeed / b3Length( w ), w );
			state->flags |= b3_isSpeedCapped;
		}

		state->linearVelocity = v;
		state->angularVelocity = w;
		state->deltaPosition = b3MulAdd( state->deltaPosition, h, v );
		state->deltaRotation = b3IntegrateRotation( state->deltaRotation, b3MulSV( h, w ) );
	}
}

static void b3IntegrateVelocitiesReference( b3BodyState* states, const b3BodySim* sims, int count, float h, b3Vec3 gravity )
{
	for ( int i = 0; i < count; ++i )
	{
		const b3BodySim* sim = sims + i;
		b3BodyState* state = states + i;
		b3Vec3 v = state->linearVelocity;
		b3Vec3 w = state->angularVelocity;

		float linearDamping = 1.0f / ( 1.0f + h * sim->linearDamping );
		float angularDamping = 1.0f / ( 1.0f + h * sim->angularDamping );
		float gravityScale = sim->invMass > 0.0f ? sim->gravityScale : 0.0f;
		b3Vec3 linearVelocityDelta = b3Blend2( h * sim->invMass, sim->force, h * gravityScale, gravity );
		v = b3MulAdd( linearVelocityDelta, linearDamping, v );
		b3Vec3 angularVelocityDelta = b3MulSV( h, b3MulMV( sim->invInertiaWorld, sim->torque ) );
		w = b3MulAdd( angularVelocityDelta, angularDamping, w );

		b3Quat q = b3MulQuat( state->deltaRotation, sim->transform.q );
		b3Matrix3 inertiaLocal = b3InvertMatrix( sim->invInertiaLocal );
		b3Vec3 omega1 = b3InvRotateVector( q, w );
		b3Vec3 omega2 = omega1;
		float i00 = inertiaLocal.cx.x;
		float i01 = inertiaLocal.cy.x;
		float i02 = inertiaLocal.cz.x;
		float i11 = inertiaLocal.cy.y;
		float i12 = inertiaLocal.cz.y;
		float i22 = inertiaLocal.cz.z;
		float w1 = omega2.x, w2 = omega2.y, w3 = omega2.z;
		float Iw1 = i00 * w1 + i01 * w2 + i02 * w3;
		float Iw2 = i01 * w1 + i11 * w2 + i12 * w3;
		float Iw3 = i02 * w1 + i12 * w2 + i22 * w3;
		b3Vec3 b = { h * ( w2 * Iw3 - w3 * Iw2 ), h * ( w3 * Iw1 - w1 * Iw3 ), h * ( w1 * Iw2 - w2 * Iw1 ) };
		b3Matrix3 J = {
			{ i00 + h * ( w2 * i02 - w3 * i01 ), i01 + h * ( w3 * i00 - w1 * i02 - Iw3 ),
			  i02 + h * ( w1 * i01 - w2 * i00 + Iw2 ) },
			{ i01 + h * ( w2 * i12 - w3 * i11 + Iw3 ), i11 + h * ( w3 * i01 - w1 * i12 ),
			  i12 + h * ( w1 * i11 - w2 * i01 - Iw1 ) },
			{ i02 + h * ( w2 * i22 - w3 * i12 - Iw2 ), i12 + h * ( w3 * i02 - w1 * i22 + Iw1 ),
			  i22 + h * ( w1 * i12 - w2 * i02 ) },
		};
		omega2 = b3Sub( omega2, b3Solve3( J, b ) );
		state->linearVelocity = v;
		state->angularVelocity = b3RotateVector( q, omega2 );
	}
}

static int MetalPositionIntegrationTest( void )
{
	const int count = 16384;
	const float h = 1.0f / 240.0f;
	const float maxLinearSpeed = 80.0f;
	const float maxAngularSpeed = 0.25f * B3_PI * 60.0f;

	b3BodyState* cpu = malloc( (size_t)count * sizeof( b3BodyState ) );
	b3BodyState* gpu = malloc( (size_t)count * sizeof( b3BodyState ) );
	ENSURE( cpu != NULL && gpu != NULL );

	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 axis = { b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) };
		axis = b3Normalize( axis );
		cpu[i] = (b3BodyState){
			.linearVelocity = { b3MetalRandomFloat( -140.0f, 140.0f ), b3MetalRandomFloat( -140.0f, 140.0f ),
								b3MetalRandomFloat( -140.0f, 140.0f ) },
			.angularVelocity = { b3MetalRandomFloat( -100.0f, 100.0f ), b3MetalRandomFloat( -100.0f, 100.0f ),
								 b3MetalRandomFloat( -100.0f, 100.0f ) },
			.deltaPosition = { b3MetalRandomFloat( -10.0f, 10.0f ), b3MetalRandomFloat( -10.0f, 10.0f ),
							   b3MetalRandomFloat( -10.0f, 10.0f ) },
			.deltaRotation = b3MakeQuatFromAxisAngle( axis, b3MetalRandomFloat( -B3_PI, B3_PI ) ),
			.flags = (uint32_t)( i % 64 ),
		};
		if ( i % 17 == 0 )
		{
			cpu[i].flags |= b3_allowFastRotation;
		}
	}
	memcpy( gpu, cpu, (size_t)count * sizeof( b3BodyState ) );

	b3IntegratePositionsReference( cpu, count, h, maxLinearSpeed, maxAngularSpeed );

	b3MetalContext* context = NULL;
	char error[1024] = { 0 };
	ENSURE( b3MetalCreateContext( &context, error, sizeof( error ) ) );
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalIntegratePositions( context, gpu, count, h, maxLinearSpeed, maxAngularSpeed, &stats ) );

	float maxError = 0.0f;
	int mismatchIndex = -1;
	for ( int i = 0; i < count; ++i )
	{
		const float* a = (const float*)( cpu + i );
		const float* b = (const float*)( gpu + i );
		for ( int j = 0; j < 13; ++j )
		{
			float errorValue = fabsf( a[j] - b[j] );
			if ( errorValue > maxError )
			{
				maxError = errorValue;
				mismatchIndex = i;
			}
		}
		ENSURE( cpu[i].flags == gpu[i].flags );
	}

	char deviceName[256];
	b3MetalGetDeviceName( context, deviceName, sizeof( deviceName ) );
	printf( "    Metal device=%s bodies=%d gpu=%.3f ms maxAbsError=%.3g at body=%d\n", deviceName, count, stats.gpuMilliseconds,
			maxError, mismatchIndex );
	ENSURE( maxError <= 3.0e-5f );

	b3MetalDestroyContext( context );
	free( gpu );
	free( cpu );
	return 0;
}

static int MetalFusedIntegrationTest( void )
{
	const int count = 8192;
	const float h = 1.0f / 240.0f;
	const float maxLinearSpeed = 80.0f;
	const float maxAngularSpeed = 0.25f * B3_PI * 60.0f;
	const b3Vec3 gravity = { 1.5f, -10.0f, 0.75f };
	b3BodyState* cpu = malloc( (size_t)count * sizeof( b3BodyState ) );
	b3BodyState* gpu = malloc( (size_t)count * sizeof( b3BodyState ) );
	b3BodySim* sims = calloc( (size_t)count, sizeof( b3BodySim ) );
	ENSURE( cpu != NULL && gpu != NULL && sims != NULL );

	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 axis = b3Normalize(
			(b3Vec3){ b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) } );
		b3Vec3 deltaAxis = b3Normalize(
			(b3Vec3){ b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) } );
		cpu[i] = (b3BodyState){
			.linearVelocity = { b3MetalRandomFloat( -120.0f, 120.0f ), b3MetalRandomFloat( -120.0f, 120.0f ),
								b3MetalRandomFloat( -120.0f, 120.0f ) },
			.angularVelocity = { b3MetalRandomFloat( -70.0f, 70.0f ), b3MetalRandomFloat( -70.0f, 70.0f ),
								 b3MetalRandomFloat( -70.0f, 70.0f ) },
			.deltaPosition = { b3MetalRandomFloat( -5.0f, 5.0f ), b3MetalRandomFloat( -5.0f, 5.0f ),
							   b3MetalRandomFloat( -5.0f, 5.0f ) },
			.deltaRotation = b3MakeQuatFromAxisAngle( deltaAxis, b3MetalRandomFloat( -0.2f, 0.2f ) ),
			.flags = (uint32_t)( i % 64 ),
		};
		if ( i % 19 == 0 )
		{
			cpu[i].flags |= b3_allowFastRotation;
		}

		float ix = b3MetalRandomFloat( 0.1f, 2.0f );
		float iy = b3MetalRandomFloat( 0.1f, 2.0f );
		float iz = b3MetalRandomFloat( 0.1f, 2.0f );
		sims[i].transform.q = b3MakeQuatFromAxisAngle( axis, b3MetalRandomFloat( -B3_PI, B3_PI ) );
		sims[i].center = (b3Pos){ b3MetalRandomFloat( -100.0f, 100.0f ), b3MetalRandomFloat( -100.0f, 100.0f ),
								  b3MetalRandomFloat( -100.0f, 100.0f ) };
		sims[i].force = (b3Vec3){ b3MetalRandomFloat( -100.0f, 100.0f ), b3MetalRandomFloat( -100.0f, 100.0f ),
								  b3MetalRandomFloat( -100.0f, 100.0f ) };
		sims[i].torque = (b3Vec3){ b3MetalRandomFloat( -30.0f, 30.0f ), b3MetalRandomFloat( -30.0f, 30.0f ),
								   b3MetalRandomFloat( -30.0f, 30.0f ) };
		sims[i].invMass = i % 23 == 0 ? 0.0f : b3MetalRandomFloat( 0.1f, 2.0f );
		sims[i].invInertiaLocal = (b3Matrix3){ { ix, 0.0f, 0.0f }, { 0.0f, iy, 0.0f }, { 0.0f, 0.0f, iz } };
		sims[i].invInertiaWorld = sims[i].invInertiaLocal;
		sims[i].linearDamping = b3MetalRandomFloat( 0.0f, 2.0f );
		sims[i].angularDamping = b3MetalRandomFloat( 0.0f, 2.0f );
		sims[i].gravityScale = b3MetalRandomFloat( -1.0f, 2.0f );
	}
	memcpy( gpu, cpu, (size_t)count * sizeof( b3BodyState ) );
	b3IntegrateVelocitiesReference( cpu, sims, count, h, gravity );
	b3IntegratePositionsReference( cpu, count, h, maxLinearSpeed, maxAngularSpeed );

	b3MetalContext* context = NULL;
	char error[1024] = { 0 };
	ENSURE( b3MetalCreateContext( &context, error, sizeof( error ) ) );
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalIntegrateUnconstrained( context, gpu, sims, count, h, gravity, maxLinearSpeed, maxAngularSpeed, &stats ) );

	float maxError = 0.0f;
	int mismatchIndex = -1;
	for ( int i = 0; i < count; ++i )
	{
		const float* a = (const float*)( cpu + i );
		const float* b = (const float*)( gpu + i );
		for ( int j = 0; j < 13; ++j )
		{
			float value = fabsf( a[j] - b[j] );
			if ( value > maxError )
			{
				maxError = value;
				mismatchIndex = i;
			}
		}
		ENSURE( cpu[i].flags == gpu[i].flags );
	}
	printf( "    fused bodies=%d gpu=%.3f ms maxAbsError=%.3g at body=%d\n", count, stats.gpuMilliseconds, maxError,
			mismatchIndex );
	ENSURE( maxError <= 1.0e-4f );

	b3MetalDestroyContext( context );
	free( sims );
	free( gpu );
	free( cpu );
	return 0;
}

static int MetalFinalizationTest( void )
{
	const int count = 4096;
	const float invTimeStep = 60.0f;
	b3BodyState* states = calloc( (size_t)count, sizeof( b3BodyState ) );
	b3BodySim* sims = calloc( (size_t)count, sizeof( b3BodySim ) );
	b3MetalFinalizeResult* reference = calloc( (size_t)count, sizeof( b3MetalFinalizeResult ) );
	ENSURE( states != NULL && sims != NULL && reference != NULL );

	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 axis = b3Normalize(
			(b3Vec3){ b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) } );
		b3Vec3 deltaAxis = b3Normalize(
			(b3Vec3){ b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) } );
		states[i].linearVelocity = (b3Vec3){ b3MetalRandomFloat( -30.0f, 30.0f ), b3MetalRandomFloat( -30.0f, 30.0f ),
											 b3MetalRandomFloat( -30.0f, 30.0f ) };
		states[i].angularVelocity = (b3Vec3){ b3MetalRandomFloat( -15.0f, 15.0f ), b3MetalRandomFloat( -15.0f, 15.0f ),
											  b3MetalRandomFloat( -15.0f, 15.0f ) };
		states[i].deltaPosition =
			(b3Vec3){ b3MetalRandomFloat( -0.2f, 0.2f ), b3MetalRandomFloat( -0.2f, 0.2f ), b3MetalRandomFloat( -0.2f, 0.2f ) };
		states[i].deltaRotation = b3MakeQuatFromAxisAngle( deltaAxis, b3MetalRandomFloat( -0.2f, 0.2f ) );
		sims[i].transform.q = b3MakeQuatFromAxisAngle( axis, b3MetalRandomFloat( -B3_PI, B3_PI ) );
		sims[i].localCenter =
			(b3Vec3){ b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ), b3MetalRandomFloat( -1.0f, 1.0f ) };
		sims[i].maxExtent =
			(b3Vec3){ b3MetalRandomFloat( 0.05f, 3.0f ), b3MetalRandomFloat( 0.05f, 3.0f ), b3MetalRandomFloat( 0.05f, 3.0f ) };
		sims[i].invInertiaLocal = (b3Matrix3){ { b3MetalRandomFloat( 0.1f, 2.0f ), 0.0f, 0.0f },
											   { 0.0f, b3MetalRandomFloat( 0.1f, 2.0f ), 0.0f },
											   { 0.0f, 0.0f, b3MetalRandomFloat( 0.1f, 2.0f ) } };

		b3Vec3 localOmega = b3InvRotateVector( sims[i].transform.q, states[i].angularVelocity );
		b3Vec3 localDelta = b3InvRotateVector( sims[i].transform.q, states[i].deltaRotation.v );
		b3Vec3 velocityArc = b3ModifiedCross( b3Abs( localOmega ), sims[i].maxExtent );
		b3Vec3 rotationArc = b3ModifiedCross( b3Abs( localDelta ), sims[i].maxExtent );
		reference[i].deltaPosition = states[i].deltaPosition;
		reference[i].rotation = b3NormalizeQuat( b3MulQuat( states[i].deltaRotation, sims[i].transform.q ) );
		reference[i].originOffset = b3Neg( b3RotateVector( reference[i].rotation, sims[i].localCenter ) );
		b3Vec3 center = { (float)sims[i].center.x, (float)sims[i].center.y, (float)sims[i].center.z };
		reference[i].transformPosition = b3Add( b3Add( center, states[i].deltaPosition ), reference[i].originOffset );
		reference[i].maxVelocity = b3Length( states[i].linearVelocity ) + b3Length( velocityArc );
		reference[i].maxDeltaPosition = b3Length( states[i].deltaPosition ) + 2.0f * b3Length( rotationArc );
		reference[i].sleepVelocity = b3MaxFloat( reference[i].maxVelocity, 0.5f * invTimeStep * reference[i].maxDeltaPosition );
		b3Matrix3 rotation = b3MakeMatrixFromQuat( reference[i].rotation );
		reference[i].invInertiaWorld = b3MulMM( b3MulMM( rotation, sims[i].invInertiaLocal ), b3Transpose( rotation ) );
	}

	b3MetalContext* context = NULL;
	char error[1024] = { 0 };
	ENSURE( b3MetalCreateContext( &context, error, sizeof( error ) ) );
	const b3MetalFinalizeResult* gpu = NULL;
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalFinalizeBodies( context, states, sims, count, invTimeStep, false, &gpu, &stats ) );
	float maxError = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		const float* a = (const float*)( reference + i );
		const float* b = (const float*)( gpu + i );
		for ( int j = 0; j < 25; ++j )
		{
			maxError = b3MaxFloat( maxError, fabsf( a[j] - b[j] ) );
		}
	}
	printf( "    finalization bodies=%d gpu=%.3f ms maxAbsError=%.3g\n", count, stats.gpuMilliseconds, maxError );
	ENSURE( maxError <= 2.0e-4f );
	b3MetalDestroyContext( context );
	free( reference );
	free( sims );
	free( states );
	return 0;
}

static bool MetalHullSphereEligible( const b3Shape* shapeA, const b3Shape* shapeB )
{
	if ( shapeA->type != b3_hullShape || shapeB->type != b3_sphereShape )
		return false;
	b3Vec3 extent = b3Sub( shapeA->hull->aabb.upperBound, shapeA->hull->aabb.lowerBound );
	float minExtent = b3MinFloat( extent.x, b3MinFloat( extent.y, extent.z ) );
	float maxExtent = b3MaxFloat( extent.x, b3MaxFloat( extent.y, extent.z ) );
	return minExtent > B3_LINEAR_SLOP && maxExtent <= 16.0f * minExtent;
}

static const b3MetalConvexManifoldResult* MetalFindConvexManifoldResult( const b3MetalConvexManifoldResult* results,
																		 int resultCount, int inputIndex )
{
	int low = 0;
	int high = resultCount;
	while ( low < high )
	{
		int middle = low + ( high - low ) / 2;
		if ( results[middle].inputIndex < (uint32_t)inputIndex )
			low = middle + 1;
		else
			high = middle;
	}
	if ( low == resultCount || results[low].inputIndex != (uint32_t)inputIndex )
		return NULL;
	return results + low;
}

static int MetalConvexManifoldTest( void )
{
#if defined( BOX3D_DOUBLE_PRECISION )
	// AABB mirrors remain float, so widely separated small deltas at 1e12 are
	// intentionally not used as independent broad-phase cells.
	const int pairCount = 1;
	const int capsuleSphereCount = 1;
	const int separatedCapsuleSphereCount = 1;
	const int crossedCapsuleCount = 1;
	const int parallelCapsuleCount = 1;
	const int hullSphereCount = 6;
#else
	const int pairCount = 32;
	const int capsuleSphereCount = 8;
	const int separatedCapsuleSphereCount = 1;
	const int crossedCapsuleCount = 8;
	const int parallelCapsuleCount = 8;
	const int hullSphereCount = 6;
#endif
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3World_SetContactRecycleDistance( worldId, 0.0f );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3Sphere sphereA = { .center = { -0.05f, 0.02f, 0.01f }, .radius = 0.55f };
	b3Sphere sphereB = { .center = { 0.04f, -0.03f, 0.02f }, .radius = 0.50f };
#if defined( BOX3D_DOUBLE_PRECISION )
	const double base = 1000000000000.0;
#else
	const float base = 0.0f;
#endif
	b3ShapeId firstSphereShape = b3_nullShapeId;
	b3ShapeId firstSphereShapeB = b3_nullShapeId;
	b3BodyId firstSphereBody = b3_nullBodyId;
	for ( int i = 0; i < pairCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_staticBody;
		bodyDef.position = (b3Pos){ base, 0.0, 4.0 * i };
		b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
		if ( i == 0 )
			firstSphereBody = bodyA;
		b3ShapeId sphereShape = b3CreateSphereShape( bodyA, &shapeDef, &sphereA );
		if ( i == 0 )
			firstSphereShape = sphereShape;
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ base + 0.72, 0.08, 4.0 * i - 0.04 };
		bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.8f, -0.2f } ), 0.15f * (float)( i % 3 ) );
		b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );
		b3ShapeId sphereShapeB = b3CreateSphereShape( bodyB, &shapeDef, &sphereB );
		if ( i == 0 )
			firstSphereShapeB = sphereShapeB;
	}
	b3SurfaceMaterial materialA = b3DefaultSurfaceMaterial();
	materialA.friction = 0.36f;
	materialA.restitution = 0.20f;
	materialA.rollingResistance = 0.40f;
	materialA.tangentVelocity = (b3Vec3){ 1.0f, 2.0f, 3.0f };
	b3SurfaceMaterial materialB = b3DefaultSurfaceMaterial();
	materialB.friction = 0.81f;
	materialB.restitution = 0.70f;
	materialB.rollingResistance = 0.20f;
	materialB.tangentVelocity = (b3Vec3){ -1.0f, 0.5f, 0.25f };
	b3Shape_SetSurfaceMaterial( firstSphereShape, materialA );
	b3Shape_SetSurfaceMaterial( firstSphereShapeB, materialB );
	int pairOffset = pairCount;
	b3Capsule capsuleA = { .center1 = { 0.0f, -0.45f, 0.0f }, .center2 = { 0.0f, 0.45f, 0.0f }, .radius = 0.30f };
	b3Capsule crossedCapsule = { .center1 = { 0.0f, 0.0f, -0.45f }, .center2 = { 0.0f, 0.0f, 0.45f }, .radius = 0.30f };
	b3Sphere capsuleSphere = { .center = { 0.02f, -0.04f, 0.01f }, .radius = 0.35f };
	for ( int i = 0; i < capsuleSphereCount; ++i )
	{
		b3BodyDef capsuleDef = b3DefaultBodyDef();
		capsuleDef.type = b3_staticBody;
		capsuleDef.position = (b3Pos){ base, 0.0, 4.0 * ( pairOffset + i ) };
		b3BodyId bodyA = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyA, &shapeDef, &capsuleA );
		capsuleDef.type = b3_dynamicBody;
		capsuleDef.position.x = base + 0.48;
		b3BodyId bodyB = b3CreateBody( worldId, &capsuleDef );
		b3CreateSphereShape( bodyB, &shapeDef, &capsuleSphere );
	}
	pairOffset += capsuleSphereCount;
	for ( int i = 0; i < separatedCapsuleSphereCount; ++i )
	{
		b3BodyDef capsuleDef = b3DefaultBodyDef();
		capsuleDef.type = b3_staticBody;
		capsuleDef.position = (b3Pos){ base, 0.0, 4.0 * ( pairOffset + i ) };
		b3BodyId bodyA = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyA, &shapeDef, &capsuleA );
		capsuleDef.type = b3_dynamicBody;
		capsuleDef.position.x = base + 0.66;
		b3BodyId bodyB = b3CreateBody( worldId, &capsuleDef );
		b3CreateSphereShape( bodyB, &shapeDef, &capsuleSphere );
	}
	pairOffset += separatedCapsuleSphereCount;
	for ( int i = 0; i < crossedCapsuleCount; ++i )
	{
		b3BodyDef capsuleDef = b3DefaultBodyDef();
		capsuleDef.type = b3_staticBody;
		capsuleDef.position = (b3Pos){ base, 0.0, 4.0 * ( pairOffset + i ) };
		b3BodyId bodyA = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyA, &shapeDef, &capsuleA );
		capsuleDef.type = b3_dynamicBody;
		capsuleDef.position.x = base + 0.42;
		b3BodyId bodyB = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyB, &shapeDef, &crossedCapsule );
	}
	pairOffset += crossedCapsuleCount;
	for ( int i = 0; i < parallelCapsuleCount; ++i )
	{
		b3BodyDef capsuleDef = b3DefaultBodyDef();
		capsuleDef.type = b3_staticBody;
		capsuleDef.position = (b3Pos){ base, 0.0, 4.0 * ( pairOffset + i ) };
		b3BodyId bodyA = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyA, &shapeDef, &capsuleA );
		capsuleDef.type = b3_dynamicBody;
		capsuleDef.position.x = base + 0.42;
		capsuleDef.position.y = 0.08;
		b3BodyId bodyB = b3CreateBody( worldId, &capsuleDef );
		b3CreateCapsuleShape( bodyB, &shapeDef, &capsuleA );
	}
	pairOffset += parallelCapsuleCount;
	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	b3ShapeId firstHullShape = b3_nullShapeId;
	b3Vec3 hullSpherePositions[6] = {
		{ 0.0f, 0.72f, 0.0f },	 { 0.65f, 0.65f, 0.0f }, { 0.62f, 0.62f, 0.62f },
		{ 0.10f, 0.20f, 0.10f }, { 0.0f, 0.81f, 0.0f },	 { 0.0f, 0.84f, 0.0f },
	};
	b3Sphere hullSphere = { .center = { 0.0f, 0.0f, 0.0f }, .radius = 0.30f };
	for ( int i = 0; i < hullSphereCount; ++i )
	{
		b3BodyDef hullDef = b3DefaultBodyDef();
		hullDef.type = b3_staticBody;
		hullDef.position = (b3Pos){ base, 0.0, 4.0 * ( pairOffset + i ) };
		hullDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, 0.7f, -0.1f } ), 0.11f * (float)i );
		b3BodyId bodyA = b3CreateBody( worldId, &hullDef );
		b3ShapeId hullShape = b3CreateHullShape( bodyA, &shapeDef, &box.base );
		if ( i == 0 )
			firstHullShape = hullShape;
		b3BodyDef sphereDef = b3DefaultBodyDef();
		sphereDef.type = b3_dynamicBody;
		sphereDef.position = hullDef.position;
		b3Vec3 worldOffset = b3RotateVector( hullDef.rotation, hullSpherePositions[i] );
		sphereDef.position.x += worldOffset.x;
		sphereDef.position.y += worldOffset.y;
		sphereDef.position.z += worldOffset.z;
		b3BodyId bodyB = b3CreateBody( worldId, &sphereDef );
		b3CreateSphereShape( bodyB, &shapeDef, &hullSphere );
	}
	pairOffset += hullSphereCount;
	b3BoxHull highAspectBox = b3MakeBoxHull( 20.0f, 0.5f, 0.5f );
	b3BodyDef highAspectDef = b3DefaultBodyDef();
	highAspectDef.type = b3_staticBody;
	highAspectDef.position = (b3Pos){ base, 0.0, 4.0 * pairOffset };
	b3BodyId highAspectA = b3CreateBody( worldId, &highAspectDef );
	b3CreateHullShape( highAspectA, &shapeDef, &highAspectBox.base );
	highAspectDef.type = b3_dynamicBody;
	highAspectDef.position.y = 0.72;
	b3BodyId highAspectB = b3CreateBody( worldId, &highAspectDef );
	b3CreateSphereShape( highAspectB, &shapeDef, &hullSphere );
	pairOffset += 1;

	// A high-aspect hull-sphere contact and a hull-hull contact prove that
	// eligibility is per record and the ordinary CPU path remains available in
	// the same narrow-phase batch.
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_staticBody;
	bodyDef.position = (b3Pos){ base, 0.0, 4.0 * pairOffset };
	b3BodyId boxA = b3CreateBody( worldId, &bodyDef );
	b3CreateHullShape( boxA, &shapeDef, &box.base );
	bodyDef.type = b3_dynamicBody;
	bodyDef.position.y = 0.6;
	b3BodyId boxB = b3CreateBody( worldId, &bodyDef );
	b3CreateHullShape( boxB, &shapeDef, &box.base );

	// Build contacts with the CPU oracle first, then exercise the batch API.
	b3World_Step( worldId, 0.0f, 1 );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	b3World* world = b3GetWorldFromId( worldId );
	int contactCount = b3GetIdCount( &world->contactIdPool );
	int expectedEligibleCount = pairCount + capsuleSphereCount + separatedCapsuleSphereCount + crossedCapsuleCount +
								parallelCapsuleCount + hullSphereCount;
	ENSURE( contactCount == expectedEligibleCount + 2 );
	int* contactIndices = malloc( (size_t)contactCount * sizeof( int ) );
	ENSURE( contactIndices != NULL );
	int cursor = 0;
	for ( int contactIndex = 0; contactIndex < world->contacts.count; ++contactIndex )
	{
		if ( world->contacts.data[contactIndex].contactId != B3_NULL_INDEX )
			contactIndices[cursor++] = contactIndex;
	}
	ENSURE( cursor == contactCount );
	b3Manifold* cpuStepManifolds = calloc( (size_t)contactCount, sizeof( b3Manifold ) );
	ENSURE( cpuStepManifolds != NULL );
	for ( int i = 0; i < contactCount; ++i )
	{
		const b3Contact* contact = world->contacts.data + contactIndices[i];
		b3ShapeType typeA = world->shapes.data[contact->shapeIdA].type;
		b3ShapeType typeB = world->shapes.data[contact->shapeIdB].type;
		bool eligible = ( typeA == b3_sphereShape && typeB == b3_sphereShape ) ||
						( typeA == b3_capsuleShape && ( typeB == b3_sphereShape || typeB == b3_capsuleShape ) ) ||
						MetalHullSphereEligible( world->shapes.data + contact->shapeIdA, world->shapes.data + contact->shapeIdB );
		if ( eligible )
		{
			ENSURE( contact->manifoldCount == 0 || contact->manifoldCount == 1 );
			if ( contact->manifoldCount == 1 )
				cpuStepManifolds[i] = contact->manifolds[0];
		}
	}

	const b3MetalConvexManifoldResult* gpu = NULL;
	int eligibleCount = 0;
	b3MetalDispatchStats stats = { 0 };
	ENSURE(
		b3MetalComputeConvexManifolds( world->metalContext, world, contactIndices, contactCount, &gpu, &eligibleCount, &stats ) );
	ENSURE( eligibleCount == expectedEligibleCount - 1 );
	ENSURE( stats.commandBufferCount == 1 );
	float maxError = 0.0f;
	int ineligibleCount = 0;
	int twoPointCount = 0;
	int separatedCount = 0;
	bool foundMaterialResult = false;
	for ( int i = 0; i < eligibleCount; ++i )
	{
		ENSURE( gpu[i].eligible == 1 );
		ENSURE( gpu[i].inputIndex < (uint32_t)contactCount );
		ENSURE( gpu[i].contactId == (uint32_t)contactIndices[gpu[i].inputIndex] );
		ENSURE( gpu[i].scanOffset == 0 );
		if ( i > 0 )
			ENSURE( gpu[i - 1].inputIndex < gpu[i].inputIndex );
	}
	b3MetalConvexManifoldResult* residentTable = malloc( (size_t)world->contacts.count * sizeof( b3MetalConvexManifoldResult ) );
	ENSURE( residentTable != NULL );
	ENSURE( b3MetalCopyResidentConvexManifoldTable( world->metalContext, residentTable, world->contacts.count - 1 ) == false );
	ENSURE( b3MetalCopyResidentConvexManifoldTable( world->metalContext, residentTable, world->contacts.count ) );
	for ( int i = 0; i < eligibleCount; ++i )
	{
		b3MetalConvexManifoldResult expected = gpu[i];
		expected.inputIndex = expected.contactId;
		ENSURE( memcmp( residentTable + expected.contactId, &expected, sizeof( expected ) ) == 0 );
	}
	for ( int i = 0; i < contactCount; ++i )
	{
		const b3MetalConvexManifoldResult* result = MetalFindConvexManifoldResult( gpu, eligibleCount, i );
		const b3Contact* contact = world->contacts.data + contactIndices[i];
		const b3Shape* shape1 = world->shapes.data + contact->shapeIdA;
		const b3Shape* shape2 = world->shapes.data + contact->shapeIdB;
		if ( ( shape1->id == firstSphereShape.index1 - 1 && shape2->id == firstSphereShapeB.index1 - 1 ) ||
			 ( shape2->id == firstSphereShape.index1 - 1 && shape1->id == firstSphereShapeB.index1 - 1 ) )
		{
			float tangentSign = shape1->id == firstSphereShape.index1 - 1 ? 1.0f : -1.0f;
			ENSURE( result != NULL );
			ENSURE( fabsf( result->friction - 0.54f ) <= 1.0e-6f );
			ENSURE( fabsf( result->restitution - 0.70f ) <= 1.0e-6f );
			ENSURE( fabsf( result->rollingResistance - 0.22f ) <= 1.0e-6f );
			ENSURE( fabsf( result->tangentVelocityX - tangentSign * 2.0f ) <= 1.0e-6f );
			ENSURE( fabsf( result->tangentVelocityY - tangentSign * 1.5f ) <= 1.0e-6f );
			ENSURE( fabsf( result->tangentVelocityZ - tangentSign * 2.75f ) <= 1.0e-6f );
			foundMaterialResult = true;
		}
		bool eligible =
			( shape1->type == b3_sphereShape && shape2->type == b3_sphereShape ) ||
			( shape1->type == b3_capsuleShape && ( shape2->type == b3_sphereShape || shape2->type == b3_capsuleShape ) ) ||
			MetalHullSphereEligible( shape1, shape2 );
		if ( eligible == false )
		{
			ENSURE( result == NULL );
			ineligibleCount += 1;
			continue;
		}
		b3Body* body1 = world->bodies.data + shape1->bodyId;
		b3Body* body2 = world->bodies.data + shape2->bodyId;
		b3Transform relative =
			b3InvMulWorldTransforms( b3GetBodyTransformQuick( world, body1 ), b3GetBodyTransformQuick( world, body2 ) );
		b3LocalManifoldPoint points[2] = { 0 };
		b3LocalManifold reference = { .points = points };
		if ( shape1->type == b3_sphereShape )
		{
			b3CollideSpheres( &reference, 2, &shape1->sphere, &shape2->sphere, relative );
		}
		else if ( shape1->type == b3_capsuleShape && shape2->type == b3_sphereShape )
		{
			b3CollideCapsuleAndSphere( &reference, 2, &shape1->capsule, &shape2->sphere, relative );
		}
		else if ( shape1->type == b3_capsuleShape && shape2->type == b3_capsuleShape )
		{
			b3CollideCapsules( &reference, 2, &shape1->capsule, &shape2->capsule, relative );
		}
		else
		{
			b3SimplexCache cache = { 0 };
			b3CollideHullAndSphere( &reference, 2, shape1->hull, &shape2->sphere, relative, &cache );
		}
		if ( shape1->type == b3_hullShape && reference.pointCount == 1 && points[0].separation > 0.0f )
		{
			// Compact hull-sphere speculative contacts remain on CPU until the
			// exact GJK normal path is available.
			ENSURE( result == NULL );
			ineligibleCount += 1;
			continue;
		}
		if ( reference.pointCount > 0 )
		{
			b3Matrix3 matrixA = b3MakeMatrixFromQuat( b3GetBodyTransformQuick( world, body1 ).q );
			reference.normal = b3MulMV( matrixA, reference.normal );
			for ( int pointIndex = 0; pointIndex < reference.pointCount; ++pointIndex )
			{
				points[pointIndex].point = b3MulMV( matrixA, points[pointIndex].point );
			}
		}
		ENSURE( result != NULL );
		ENSURE( result->eligible == 1 );
		ENSURE( result->touching == (uint32_t)( reference.pointCount > 0 ) );
		ENSURE( result->pointCount == (uint32_t)reference.pointCount );
		if ( reference.pointCount == 0 )
			separatedCount += 1;
		if ( reference.pointCount == 2 )
			twoPointCount += 1;
		if ( reference.pointCount > 0 )
		{
			maxError = b3MaxFloat( maxError, fabsf( result->normalX - reference.normal.x ) );
			maxError = b3MaxFloat( maxError, fabsf( result->normalY - reference.normal.y ) );
			maxError = b3MaxFloat( maxError, fabsf( result->normalZ - reference.normal.z ) );
			float gpuPointX[2] = { result->point1X, result->point2X };
			float gpuPointY[2] = { result->point1Y, result->point2Y };
			float gpuPointZ[2] = { result->point1Z, result->point2Z };
			float gpuSeparation[2] = { result->separation1, result->separation2 };
			uint32_t gpuFeatureId[2] = { result->featureId1, result->featureId2 };
			float gpuAnchorBX[2] = { result->anchorB1X, result->anchorB2X };
			float gpuAnchorBY[2] = { result->anchorB1Y, result->anchorB2Y };
			float gpuAnchorBZ[2] = { result->anchorB1Z, result->anchorB2Z };
			const b3Manifold* cpuManifold = cpuStepManifolds + i;
			ENSURE( cpuManifold->pointCount == reference.pointCount );
			for ( int pointIndex = 0; pointIndex < reference.pointCount; ++pointIndex )
			{
				maxError = b3MaxFloat( maxError, fabsf( gpuPointX[pointIndex] - cpuManifold->points[pointIndex].anchorA.x ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuPointY[pointIndex] - cpuManifold->points[pointIndex].anchorA.y ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuPointZ[pointIndex] - cpuManifold->points[pointIndex].anchorA.z ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuAnchorBX[pointIndex] - cpuManifold->points[pointIndex].anchorB.x ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuAnchorBY[pointIndex] - cpuManifold->points[pointIndex].anchorB.y ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuAnchorBZ[pointIndex] - cpuManifold->points[pointIndex].anchorB.z ) );
				maxError = b3MaxFloat( maxError, fabsf( gpuSeparation[pointIndex] - points[pointIndex].separation ) );
				ENSURE( gpuFeatureId[pointIndex] == b3MakeFeatureId( points[pointIndex].pair ) );
			}
		}
	}
	ENSURE( ineligibleCount == 3 );
	ENSURE( foundMaterialResult );
	ENSURE( twoPointCount == parallelCapsuleCount );
	ENSURE( separatedCount == separatedCapsuleSphereCount + 1 );
	ENSURE( maxError <= 3.0e-5f );

	int firstEligibleCount = eligibleCount;
	b3MetalConvexManifoldResult* first = malloc( (size_t)firstEligibleCount * sizeof( b3MetalConvexManifoldResult ) );
	ENSURE( first != NULL );
	memcpy( first, gpu, (size_t)firstEligibleCount * sizeof( b3MetalConvexManifoldResult ) );
	ENSURE(
		b3MetalComputeConvexManifolds( world->metalContext, world, contactIndices, contactCount, &gpu, &eligibleCount, &stats ) );
	ENSURE( eligibleCount == firstEligibleCount );
	ENSURE( memcmp( first, gpu, (size_t)firstEligibleCount * sizeof( b3MetalConvexManifoldResult ) ) == 0 );
	for ( int i = 0; i < contactCount / 2; ++i )
	{
		B3_SWAP( contactIndices[i], contactIndices[contactCount - 1 - i] );
	}
	ENSURE(
		b3MetalComputeConvexManifolds( world->metalContext, world, contactIndices, contactCount, &gpu, &eligibleCount, &stats ) );
	ENSURE( eligibleCount == firstEligibleCount );
	b3MetalConvexManifoldResult* reorderedTable = malloc( (size_t)world->contacts.count * sizeof( b3MetalConvexManifoldResult ) );
	ENSURE( reorderedTable != NULL );
	ENSURE( b3MetalCopyResidentConvexManifoldTable( world->metalContext, reorderedTable, world->contacts.count ) );
	for ( int i = 0; i < firstEligibleCount; ++i )
	{
		uint32_t contactId = first[i].contactId;
		ENSURE( memcmp( residentTable + contactId, reorderedTable + contactId, sizeof( b3MetalConvexManifoldResult ) ) == 0 );
	}
	for ( int i = 0; i < contactCount / 2; ++i )
	{
		B3_SWAP( contactIndices[i], contactIndices[contactCount - 1 - i] );
	}

	b3World_Step( worldId, 0.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	float maxApplyError = 0.0f;
	for ( int i = 0; i < contactCount; ++i )
	{
		b3Contact* contact = world->contacts.data + contactIndices[i];
		b3ShapeType typeA = world->shapes.data[contact->shapeIdA].type;
		b3ShapeType typeB = world->shapes.data[contact->shapeIdB].type;
		bool eligible = ( typeA == b3_sphereShape && typeB == b3_sphereShape ) ||
						( typeA == b3_capsuleShape && ( typeB == b3_sphereShape || typeB == b3_capsuleShape ) ) ||
						MetalHullSphereEligible( world->shapes.data + contact->shapeIdA, world->shapes.data + contact->shapeIdB );
		if ( eligible == false )
			continue;
		const b3Manifold* cpu = cpuStepManifolds + i;
		ENSURE( contact->manifoldCount == ( cpu->pointCount > 0 ? 1 : 0 ) );
		if ( contact->manifoldCount == 0 )
			continue;
		const b3Manifold* applied = contact->manifolds;
		ENSURE( cpu->pointCount == applied->pointCount );
		maxApplyError = b3MaxFloat( maxApplyError, b3Length( b3Sub( cpu->normal, applied->normal ) ) );
		for ( int pointIndex = 0; pointIndex < cpu->pointCount; ++pointIndex )
		{
			maxApplyError = b3MaxFloat(
				maxApplyError, b3Length( b3Sub( cpu->points[pointIndex].anchorA, applied->points[pointIndex].anchorA ) ) );
			maxApplyError = b3MaxFloat(
				maxApplyError, b3Length( b3Sub( cpu->points[pointIndex].anchorB, applied->points[pointIndex].anchorB ) ) );
			maxApplyError =
				b3MaxFloat( maxApplyError, fabsf( cpu->points[pointIndex].separation - applied->points[pointIndex].separation ) );
			ENSURE( cpu->points[pointIndex].featureId == applied->points[pointIndex].featureId );
		}
	}
	printf( "    convex manifolds contacts=%d eligible=%d separated=%d twoPoint=%d VF64=%s gpu=%.3f ms oracleError=%.3g "
			"applyError=%.3g deterministic=yes\n",
			contactCount, eligibleCount, separatedCount, twoPointCount,
#if defined( BOX3D_DOUBLE_PRECISION )
			"yes",
#else
			"no",
#endif
			stats.gpuMilliseconds, maxError, maxApplyError );
	ENSURE( profile.narrowPhaseDispatchCount == 1 );
	ENSURE( profile.narrowPhaseFallbackCount == 0 );
	ENSURE( profile.narrowPhaseGeometryUploadCount == 1 );
	ENSURE( profile.narrowPhaseGeometryReuseCount >= 2 );
	ENSURE( profile.narrowPhaseTransformUploadCount == 1 );
	ENSURE( profile.narrowPhaseTransformReuseCount >= 2 );
	ENSURE( profile.lastNarrowPhaseHullShapeCount == 8 );
	ENSURE( profile.lastNarrowPhaseUniqueHullCount == 1 );
	ENSURE( profile.lastNarrowPhaseResultCount == expectedEligibleCount - 1 );
	ENSURE( profile.lastNarrowPhaseManifoldTableCount == world->contacts.count );
	printf( "    resident narrow inputs geometry=%llu/%llu transforms=%llu/%llu hullShapes=%d uniqueHulls=%d inputBytes=32 "
			"resultBytes=%zu tableSlots=%d\n",
			(unsigned long long)profile.narrowPhaseGeometryUploadCount, (unsigned long long)profile.narrowPhaseGeometryReuseCount,
			(unsigned long long)profile.narrowPhaseTransformUploadCount,
			(unsigned long long)profile.narrowPhaseTransformReuseCount, profile.lastNarrowPhaseHullShapeCount,
			profile.lastNarrowPhaseUniqueHullCount,
			(size_t)profile.lastNarrowPhaseResultCount * sizeof( b3MetalConvexManifoldResult ), world->contacts.count );
	ENSURE( maxApplyError <= 5.0e-5f );

	free( cpuStepManifolds );
	free( first );
	free( residentTable );
	free( reorderedTable );
	free( contactIndices );
	b3BoxHull replacementBox = b3MakeBoxHull( 0.55f, 0.50f, 0.50f );
	b3Shape_SetHull( firstHullShape, &replacementBox.base );
	b3World_Step( worldId, 0.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseGeometryUploadCount == 2 );
	ENSURE( profile.lastNarrowPhaseHullShapeCount == 8 );
	ENSURE( profile.lastNarrowPhaseUniqueHullCount == 2 );
	printf( "    resident hull mutation uploads=%llu shapes=%d unique=%d rebuild=yes\n",
			(unsigned long long)profile.narrowPhaseGeometryUploadCount, profile.lastNarrowPhaseHullShapeCount,
			profile.lastNarrowPhaseUniqueHullCount );
	b3Sphere replacementSphere = sphereA;
	replacementSphere.center.x += 0.01f;
	b3Shape_SetSphere( firstSphereShape, &replacementSphere );
	b3World_Step( worldId, 0.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseGeometryUploadCount == 3 );
	printf( "    resident primitive mutation uploads=%llu rebuild=yes\n",
			(unsigned long long)profile.narrowPhaseGeometryUploadCount );
	b3Body_SetTransform( firstSphereBody, (b3Pos){ base + 0.01, 0.0, 0.0 }, b3Quat_identity );
	b3World_Step( worldId, 0.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseTransformUploadCount == 2 );
	ENSURE( profile.narrowPhaseGeometryUploadCount == 3 );
	printf( "    resident transform mutation uploads=%llu geometryStable=yes rebuild=yes\n",
			(unsigned long long)profile.narrowPhaseTransformUploadCount );
	for ( int step = 0; step < 3; ++step )
		b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseTransformUploadCount == 4 );
	ENSURE( profile.narrowPhaseGeometryUploadCount == 3 );
	printf( "    resident transform cadence solvedSteps=3 uploads=%llu geometryStable=yes\n",
			(unsigned long long)profile.narrowPhaseTransformUploadCount );
	uint64_t geometryUploadsBeforeMaterialChange = profile.narrowPhaseGeometryUploadCount;
	b3Shape_SetFriction( firstSphereShape, 0.49f );
	b3World_Step( worldId, 0.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseGeometryUploadCount == geometryUploadsBeforeMaterialChange + 1 );
	printf( "    resident material mutation geometryUploads=%llu rebuild=yes\n",
			(unsigned long long)profile.narrowPhaseGeometryUploadCount );
	b3DestroyWorld( worldId );
	return 0;
}

static int MetalPairTraversalTest( void )
{
	const int bodyCount = 607;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.46f };
	b3ShapeId mutableShapeId = { 0 };
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = i % 4 == 0 ? b3_staticBody : i % 4 == 1 ? b3_kinematicBody : b3_dynamicBody;
		bodyDef.position = (b3Pos){ 0.62f * (float)( i % 16 ), 0.62f * (float)( ( i / 16 ) % 8 ), 0.62f * (float)( i / 128 ) };
		b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		if ( i % 16 == 2 || i % 16 == 3 )
			shapeDef.filter.groupIndex = -2;
		if ( i % 16 == 6 || i % 16 == 7 )
		{
			shapeDef.filter.groupIndex = 3;
			shapeDef.filter.maskBits = 0;
		}
		if ( i % 16 == 10 )
			shapeDef.isSensor = true;
		if ( i % 16 == 11 )
			shapeDef.filter.maskBits = 0;
		b3ShapeId shapeId = b3CreateSphereShape( bodyId, &shapeDef, &sphere );
		if ( i == 12 )
			mutableShapeId = shapeId;
		if ( i % 50 == 0 )
		{
			b3ShapeDef extraShapeDef = b3DefaultShapeDef();
			b3CreateSphereShape( bodyId, &extraShapeDef, &sphere );
		}
	}

	b3World* world = b3GetWorldFromId( worldId );
	b3BroadPhase* broadPhase = &world->broadPhase;
	int moveCount = broadPhase->moveArray.count;
	ENSURE( moveCount == bodyCount + ( bodyCount + 49 ) / 50 );
	const b3MetalPairQueryRecord* gpuRecords = NULL;
	const b3MetalPairCandidate* gpuCandidates = NULL;
	int gpuCandidateCount = 0;
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, moveCount, &gpuRecords,
										   &gpuCandidates, &gpuCandidateCount, &stats ) );
	ENSURE( stats.commandBufferCount == 1 );
	ENSURE( stats.treeUploadCount == 1 );
	ENSURE( stats.metadataUploadCount == 1 );
	ENSURE( stats.pairSetUploadCount == 1 );
	double initialGpuMilliseconds = stats.gpuMilliseconds;
	stats = (b3MetalDispatchStats){ 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, moveCount, &gpuRecords,
										   &gpuCandidates, &gpuCandidateCount, &stats ) );
	ENSURE( stats.commandBufferCount == 1 );
	ENSURE( stats.treeUploadCount == 0 );
	ENSURE( stats.metadataUploadCount == 0 );
	ENSURE( stats.pairSetUploadCount == 0 );
	double steadyGpuMilliseconds = stats.gpuMilliseconds;

	int cpuCapacity = bodyCount * bodyCount;
	b3MetalPairCandidate* cpuCandidates = malloc( (size_t)cpuCapacity * sizeof( b3MetalPairCandidate ) );
	ENSURE( cpuCandidates != NULL );
	b3PairCandidateCapture capture = {
		.candidates = cpuCandidates,
		.capacity = cpuCapacity,
		.broadPhase = broadPhase,
		.world = world,
	};
	int totalCpuCount = 0;
	for ( int moveIndex = 0; moveIndex < moveCount; ++moveIndex )
	{
		int proxyKey = broadPhase->moveArray.data[moveIndex];
		capture.queryProxyKey = proxyKey;
		b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
		int proxyId = B3_PROXY_ID( proxyKey );
		b3AABB fatAABB = b3DynamicTree_GetAABB( broadPhase->trees + proxyType, proxyId );
		ENSURE( gpuRecords[moveIndex].queryShapeIndex ==
				(int)b3DynamicTree_GetUserData( broadPhase->trees + proxyType, proxyId ) );
		capture.queryShapeIndex = gpuRecords[moveIndex].queryShapeIndex;
		ENSURE( gpuRecords[moveIndex].lowerX == fatAABB.lowerBound.x );
		ENSURE( gpuRecords[moveIndex].lowerY == fatAABB.lowerBound.y );
		ENSURE( gpuRecords[moveIndex].lowerZ == fatAABB.lowerBound.z );
		ENSURE( gpuRecords[moveIndex].upperX == fatAABB.upperBound.x );
		ENSURE( gpuRecords[moveIndex].upperY == fatAABB.upperBound.y );
		ENSURE( gpuRecords[moveIndex].upperZ == fatAABB.upperBound.z );
		capture.count = 0;
		if ( proxyType == b3_dynamicBody )
		{
			capture.treeType = b3_kinematicBody;
			b3DynamicTree_Query( broadPhase->trees + b3_kinematicBody, fatAABB, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
								 &capture );
			capture.treeType = b3_staticBody;
			b3DynamicTree_Query( broadPhase->trees + b3_staticBody, fatAABB, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
								 &capture );
		}
		capture.treeType = b3_dynamicBody;
		b3DynamicTree_Query( broadPhase->trees + b3_dynamicBody, fatAABB, B3_DEFAULT_MASK_BITS, false, CapturePairCandidate,
							 &capture );

		ENSURE( gpuRecords[moveIndex].flags == 0 );
		ENSURE( gpuRecords[moveIndex].count == (uint32_t)capture.count );
		for ( int candidateIndex = 0; candidateIndex < capture.count; ++candidateIndex )
		{
			const b3MetalPairCandidate* cpu = cpuCandidates + candidateIndex;
			const b3MetalPairCandidate* gpu = gpuCandidates + gpuRecords[moveIndex].offset + candidateIndex;
			ENSURE( cpu->proxyId == gpu->proxyId );
			ENSURE( cpu->treeType == gpu->treeType );
			ENSURE( cpu->shapeIndex == gpu->shapeIndex );
		}
		totalCpuCount += capture.count;
	}
	ENSURE( totalCpuCount == gpuCandidateCount );
	int dynamicProxyKey = B3_NULL_INDEX;
	for ( int moveIndex = 0; moveIndex < moveCount; ++moveIndex )
	{
		if ( B3_PROXY_TYPE( broadPhase->moveArray.data[moveIndex] ) == b3_dynamicBody )
		{
			dynamicProxyKey = broadPhase->moveArray.data[moveIndex];
			break;
		}
	}
	ENSURE( dynamicProxyKey != B3_NULL_INDEX );
	b3Filter disabledFilter = b3DefaultFilter();
	disabledFilter.maskBits = 0;
	b3Shape_SetFilter( mutableShapeId, disabledFilter, false );
	int dynamicProxyId = B3_PROXY_ID( dynamicProxyKey );
	b3AABB expandedAABB = b3DynamicTree_GetAABB( broadPhase->trees + b3_dynamicBody, dynamicProxyId );
	expandedAABB.lowerBound.x -= 0.01f;
	expandedAABB.upperBound.x += 0.01f;
	b3BroadPhase_EnlargeProxy( broadPhase, dynamicProxyKey, expandedAABB );
	stats = (b3MetalDispatchStats){ 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, moveCount, &gpuRecords,
										   &gpuCandidates, &gpuCandidateCount, &stats ) );
	ENSURE( stats.treeUploadCount == 1 );
	ENSURE( stats.metadataUploadCount == 1 );
	ENSURE( stats.pairSetUploadCount == 0 );
	ENSURE( VerifyResidentPairTraversal( world ) == 0 );
	printf( "    pair traversal moves=%d candidates=%d initial=%.3f ms steady=%.3f ms exactOrder=yes\n", moveCount,
			gpuCandidateCount, initialGpuMilliseconds, steadyGpuMilliseconds );

	free( cpuCandidates );
	b3DestroyWorld( worldId );
	return 0;
}

static int MetalResidentSolverOwnershipTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorldId = b3CreateWorld( &worldDef );
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	b3World_SetContactRecycleDistance( worldId, 0.0f );

	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId cpuBodyA = b3CreateBody( cpuWorldId, &bodyDef );
	b3CreateSphereShape( cpuBodyA, &shapeDef, &sphere );
	b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( bodyA, &shapeDef, &sphere );
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.position = (b3Pos){ 0.99, 0.0, 0.0 };
	b3BodyId cpuBodyB = b3CreateBody( cpuWorldId, &bodyDef );
	b3CreateSphereShape( cpuBodyB, &shapeDef, &sphere );
	b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( bodyB, &shapeDef, &sphere );

	b3World_Step( cpuWorldId, 1.0f / 60.0f, 1 );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.lastNarrowPhaseResultCount == 1 );
	ENSURE( profile.lastResidentConvexContactCount == 1 );
	ENSURE( profile.lastResidentConvexConstraintCount == 1 );
	ENSURE( profile.contactPrepareDispatchCount == 1 );
	ENSURE( profile.contactPrepareFallbackCount == 0 );
	ENSURE( profile.lastContactPrepareIndexBytes == 4 * sizeof( uint32_t ) );
	ENSURE( profile.lastContactImpulseResultBytes == sizeof( b3MetalContactImpulseResult ) );
	ENSURE( profile.contactSchedulePackCount == 1 );
	ENSURE( profile.contactScheduleReuseCount == 0 );
	b3Vec3 cpuVelocity = b3Body_GetLinearVelocity( cpuBodyB );
	b3Vec3 gpuVelocity = b3Body_GetLinearVelocity( bodyB );
	ENSURE( b3Length( b3Sub( cpuVelocity, gpuVelocity ) ) <= 3.0e-5f );

	// The next unchanged step deliberately takes the CPU recycling exception.
	// A table slot still exists, but it must no longer authorize GPU preparation.
	b3World_SetContactRecycleDistance( worldId, 0.1f );
	b3World_SetContactRecycleDistance( cpuWorldId, 0.1f );
	b3World_Step( cpuWorldId, 1.0f / 60.0f, 1 );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.lastNarrowPhaseResultCount == 1 );
	ENSURE( profile.lastResidentConvexContactCount == 0 );
	ENSURE( profile.lastResidentConvexConstraintCount == 0 );
	ENSURE( profile.contactPrepareDispatchCount == 1 );
	ENSURE( profile.contactPrepareFallbackCount == 0 );
	ENSURE( profile.lastContactPrepareIndexBytes == 0 );
	ENSURE( profile.lastContactImpulseResultBytes == 0 );
	ENSURE( profile.contactSchedulePackCount == 1 );
	ENSURE( profile.contactScheduleReuseCount == 0 );

	b3DestroyWorld( worldId );
	b3DestroyWorld( cpuWorldId );
	return 0;
}

static bool MetalPreparePreSolveCallback( b3ShapeId shapeIdA, b3ShapeId shapeIdB, b3Pos point, b3Vec3 normal, void* context )
{
	B3_UNUSED( shapeIdA, shapeIdB, point, normal );
	*(int*)context += 1;
	return true;
}

static float MetalCustomFrictionCallback( float frictionA, uint64_t materialA, float frictionB, uint64_t materialB )
{
	B3_UNUSED( frictionA, materialA, frictionB, materialB );
	return 0.123f;
}

static float MetalCustomRestitutionCallback( float restitutionA, uint64_t materialA, float restitutionB, uint64_t materialB )
{
	B3_UNUSED( restitutionA, materialA, restitutionB, materialB );
	return 0.456f;
}

static int MetalContactMaterialCallbackExceptionTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	worldDef.frictionCallback = MetalCustomFrictionCallback;
	worldDef.restitutionCallback = MetalCustomRestitutionCallback;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	b3World_SetContactRecycleDistance( worldId, 0.0f );
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
	b3ShapeId shapeA = b3CreateSphereShape( bodyA, &shapeDef, &sphere );
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.position.x = 0.8;
	b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );
	b3ShapeId shapeB = b3CreateSphereShape( bodyB, &shapeDef, &sphere );
	b3SurfaceMaterial materialA = b3DefaultSurfaceMaterial();
	materialA.rollingResistance = 0.4f;
	materialA.tangentVelocity = (b3Vec3){ 1.0f, 2.0f, 3.0f };
	b3SurfaceMaterial materialB = b3DefaultSurfaceMaterial();
	materialB.rollingResistance = 0.2f;
	materialB.tangentVelocity = (b3Vec3){ -1.0f, 0.5f, 0.25f };
	b3Shape_SetSurfaceMaterial( shapeA, materialA );
	b3Shape_SetSurfaceMaterial( shapeB, materialB );
	b3World_Step( worldId, 0.0f, 1 );
	b3World* world = b3GetWorldFromId( worldId );
	ENSURE( b3GetIdCount( &world->contactIdPool ) == 1 );
	b3Contact* contact = NULL;
	for ( int contactIndex = 0; contactIndex < world->contacts.count; ++contactIndex )
	{
		if ( world->contacts.data[contactIndex].contactId != B3_NULL_INDEX )
			contact = world->contacts.data + contactIndex;
	}
	ENSURE( contact != NULL && contact->manifoldCount == 1 );
	ENSURE( contact->friction == 0.123f );
	ENSURE( contact->restitution == 0.456f );
	ENSURE( fabsf( contact->rollingResistance - 0.2f ) <= 1.0e-6f );
	float tangentSign = contact->shapeIdA == shapeA.index1 - 1 ? 1.0f : -1.0f;
	ENSURE( b3Length( b3Sub( contact->tangentVelocity,
							 (b3Vec3){ tangentSign * 2.0f, tangentSign * 1.5f, tangentSign * 2.75f } ) ) <= 1.0e-6f );
	b3World_Step( worldId, 0.0f, 1 );
	ENSURE( contact->friction == 0.123f );
	ENSURE( contact->restitution == 0.456f );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.narrowPhaseDispatchCount == 2 );
	ENSURE( profile.contactPrepareDeviceRefreshCount == 0 );
	printf( "    material callbacks custom=cpu rolling+tangent=gpu narrowDispatches=2 deviceRefreshes=0\n" );
	b3DestroyWorld( worldId );
	return 0;
}

static int MetalContactPreparePreSolveExceptionTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	b3World_SetContactRecycleDistance( worldId, 0.0f );
	int callbackCount = 0;
	b3World_SetPreSolveCallback( worldId, MetalPreparePreSolveCallback, &callbackCount );

	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( bodyA, &shapeDef, &sphere );
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.position = (b3Pos){ 0.99, 0.0, 0.0 };
	b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );
	shapeDef.enablePreSolveEvents = true;
	b3CreateSphereShape( bodyB, &shapeDef, &sphere );

	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	ENSURE( callbackCount == 1 );
	ENSURE( profile.lastNarrowPhaseResultCount == 1 );
	ENSURE( profile.lastResidentConvexContactCount == 0 );
	ENSURE( profile.lastResidentConvexConstraintCount == 0 );
	ENSURE( profile.contactPrepareDispatchCount == 0 );
	ENSURE( profile.contactPrepareFallbackCount == 0 );
	ENSURE( profile.lastContactPrepareIndexBytes == 0 );

	b3DestroyWorld( worldId );
	return 0;
}

static int MetalExistingPairFilterTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	ENSURE( b3World_SetMetalBroadPhase( worldId, true ) );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.position = (b3Pos){ -0.25f, 0.0f, 0.0f };
	b3BodyId bodyA = b3CreateBody( worldId, &bodyDef );
	b3ShapeId shapeA = b3CreateSphereShape( bodyA, &shapeDef, &sphere );
	bodyDef.position = (b3Pos){ 0.25f, 0.0f, 0.0f };
	b3BodyId bodyB = b3CreateBody( worldId, &bodyDef );
	b3ShapeId shapeB = b3CreateSphereShape( bodyB, &shapeDef, &sphere );
	b3World* world = b3GetWorldFromId( worldId );
	b3BroadPhase* broadPhase = &world->broadPhase;
	const b3MetalPairQueryRecord* records = NULL;
	const b3MetalPairCandidate* candidates = NULL;
	int candidateCount = 0;
	b3MetalDispatchStats stats = { 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, broadPhase->moveArray.count,
										   &records, &candidates, &candidateCount, &stats ) );
	ENSURE( candidateCount == 1 );
	ENSURE( stats.pairSetUploadCount == 1 );

	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	ENSURE( broadPhase->pairSet.count == 1 );
	int shapeIndexA = shapeA.index1 - 1;
	int shapeIndexB = shapeB.index1 - 1;
	b3BufferMove( broadPhase, world->shapes.data[shapeIndexA].proxyKey );
	b3BufferMove( broadPhase, world->shapes.data[shapeIndexB].proxyKey );
	stats = (b3MetalDispatchStats){ 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, broadPhase->moveArray.count,
										   &records, &candidates, &candidateCount, &stats ) );
	ENSURE( candidateCount == 0 );
	ENSURE( stats.pairSetUploadCount == 1 );
	ENSURE( VerifyResidentPairTraversal( world ) == 0 );

	b3Filter disabledFilter = b3DefaultFilter();
	disabledFilter.maskBits = 0;
	b3Shape_SetFilter( shapeB, disabledFilter, true );
	ENSURE( broadPhase->pairSet.count == 0 );
	b3Shape_SetFilter( shapeB, b3DefaultFilter(), false );
	b3BufferMove( broadPhase, world->shapes.data[shapeIndexA].proxyKey );
	b3BufferMove( broadPhase, world->shapes.data[shapeIndexB].proxyKey );
	stats = (b3MetalDispatchStats){ 0 };
	ENSURE( b3MetalGeneratePairCandidates( world->metalContext, world, broadPhase->moveArray.data, broadPhase->moveArray.count,
										   &records, &candidates, &candidateCount, &stats ) );
	ENSURE( candidateCount == 1 );
	ENSURE( stats.pairSetUploadCount == 1 );
	ENSURE( stats.metadataUploadCount == 1 );
	printf( "    existing pair filter contacts=1 suppressed=1 restored=%d uploads=stable\n", candidateCount );

	b3DestroyWorld( worldId );
	return 0;
}

static int MetalContactPrepareFallbackTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3World_SetContactRecycleDistance( cpuWorld, 0.0f );
	b3World_SetContactRecycleDistance( gpuWorld, 0.0f );

	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId cpuStatic = b3CreateBody( cpuWorld, &bodyDef );
	b3BodyId gpuStatic = b3CreateBody( gpuWorld, &bodyDef );
	b3CreateSphereShape( cpuStatic, &shapeDef, &sphere );
	b3CreateSphereShape( gpuStatic, &shapeDef, &sphere );
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.position = (b3Pos){ 0.99, 0.0, 0.0 };
	b3BodyId cpuContactBody = b3CreateBody( cpuWorld, &bodyDef );
	b3BodyId gpuContactBody = b3CreateBody( gpuWorld, &bodyDef );
	b3CreateSphereShape( cpuContactBody, &shapeDef, &sphere );
	b3CreateSphereShape( gpuContactBody, &shapeDef, &sphere );

	bodyDef.position = (b3Pos){ -0.5, 5.0, 0.0 };
	b3BodyId cpuJointA = b3CreateBody( cpuWorld, &bodyDef );
	b3BodyId gpuJointA = b3CreateBody( gpuWorld, &bodyDef );
	bodyDef.position = (b3Pos){ 0.5, 5.0, 0.0 };
	bodyDef.linearVelocity = (b3Vec3){ 0.0f, 0.2f, 0.0f };
	b3BodyId cpuJointB = b3CreateBody( cpuWorld, &bodyDef );
	b3BodyId gpuJointB = b3CreateBody( gpuWorld, &bodyDef );
	b3RevoluteJointDef cpuJointDef = b3DefaultRevoluteJointDef();
	cpuJointDef.base.bodyIdA = cpuJointA;
	cpuJointDef.base.bodyIdB = cpuJointB;
	b3CreateRevoluteJoint( cpuWorld, &cpuJointDef );
	b3RevoluteJointDef gpuJointDef = cpuJointDef;
	gpuJointDef.base.bodyIdA = gpuJointA;
	gpuJointDef.base.bodyIdB = gpuJointB;
	b3CreateRevoluteJoint( gpuWorld, &gpuJointDef );

	b3World_Step( cpuWorld, 1.0f / 60.0f, 1 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	ENSURE( profile.lastResidentConvexConstraintCount == 1 );
	ENSURE( profile.contactPrepareDispatchCount == 0 );
	ENSURE( profile.contactPrepareFallbackCount == 1 );
	ENSURE( profile.lastContactPrepareIndexBytes == 0 );
	ENSURE( profile.jointFallbackCount == 1 );
	b3World* gpu = b3GetWorldFromId( gpuWorld );
	int staleResultCount = 0;
	ENSURE( b3MetalGetResidentContactImpulseTable( gpu->metalContext, NULL, &staleResultCount ) == NULL );
	ENSURE( staleResultCount == 0 );
	ENSURE( b3Length( b3Sub( b3Body_GetLinearVelocity( cpuContactBody ), b3Body_GetLinearVelocity( gpuContactBody ) ) ) <=
			3.0e-5f );
	ENSURE( b3Length( b3Sub( b3Body_GetLinearVelocity( cpuJointB ), b3Body_GetLinearVelocity( gpuJointB ) ) ) <= 3.0e-5f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalResidentContactPrepareDifferentialTest( void )
{
	const int sphereCount = 64;
	const int capsuleCount = 17;
	const int count = sphereCount + capsuleCount;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3World_SetContactRecycleDistance( cpuWorld, 0.0f );
	b3World_SetContactRecycleDistance( gpuWorld, 0.0f );
	b3ShapeDef staticShapeDef = b3DefaultShapeDef();
	staticShapeDef.baseMaterial.friction = 0.65f;
	staticShapeDef.baseMaterial.restitution = 0.35f;
	staticShapeDef.baseMaterial.rollingResistance = 0.08f;
	staticShapeDef.baseMaterial.tangentVelocity = (b3Vec3){ 0.0f, 0.12f, -0.07f };
	b3ShapeDef dynamicShapeDef = staticShapeDef;
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3Capsule capsule = { .center1 = { 0.0f, -0.4f, 0.0f }, .center2 = { 0.0f, 0.4f, 0.0f }, .radius = 0.3f };
	b3BodyId cpuBodies[count];
	b3BodyId gpuBodies[count];
	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef staticDef = b3DefaultBodyDef();
		staticDef.position = (b3Pos){ 0.0, 0.0, 3.0 * (double)i };
		b3BodyId cpuStatic = b3CreateBody( cpuWorld, &staticDef );
		b3BodyId gpuStatic = b3CreateBody( gpuWorld, &staticDef );
		b3BodyDef dynamicDef = b3DefaultBodyDef();
		dynamicDef.type = b3_dynamicBody;
		dynamicDef.enableSleep = false;
		dynamicDef.position = (b3Pos){ i < sphereCount ? 0.96 : 0.56, 0.0, 3.0 * (double)i };
		dynamicDef.linearVelocity = (b3Vec3){ -0.4f - 0.002f * (float)i, 0.03f * (float)( i % 3 - 1 ), 0.0f };
		dynamicDef.angularVelocity = (b3Vec3){ 0.02f * (float)( i % 5 ), -0.03f, 0.04f };
		cpuBodies[i] = b3CreateBody( cpuWorld, &dynamicDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &dynamicDef );
		if ( i < sphereCount )
		{
			b3CreateSphereShape( cpuStatic, &staticShapeDef, &sphere );
			b3CreateSphereShape( gpuStatic, &staticShapeDef, &sphere );
			b3CreateSphereShape( cpuBodies[i], &dynamicShapeDef, &sphere );
			b3CreateSphereShape( gpuBodies[i], &dynamicShapeDef, &sphere );
		}
		else
		{
			b3CreateCapsuleShape( cpuStatic, &staticShapeDef, &capsule );
			b3CreateCapsuleShape( gpuStatic, &staticShapeDef, &capsule );
			b3CreateCapsuleShape( cpuBodies[i], &dynamicShapeDef, &capsule );
			b3CreateCapsuleShape( gpuBodies[i], &dynamicShapeDef, &capsule );
		}
	}

	for ( int step = 0; step < 4; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, b3Length( b3SubPos( a.p, b.p ) ) );
		maxTransformError = b3MaxFloat( maxTransformError, b3Length( b3Sub( a.q.v, b.q.v ) ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		maxVelocityError =
			b3MaxFloat( maxVelocityError,
						b3Length( b3Sub( b3Body_GetLinearVelocity( cpuBodies[i] ), b3Body_GetLinearVelocity( gpuBodies[i] ) ) ) );
		maxVelocityError = b3MaxFloat( maxVelocityError, b3Length( b3Sub( b3Body_GetAngularVelocity( cpuBodies[i] ),
																		  b3Body_GetAngularVelocity( gpuBodies[i] ) ) ) );
	}
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    resident contact prepare contacts=%d dispatches=%llu deviceRefreshes=%llu schedule=%llu/%llu indexBytes=%llu "
			"legacyBytes=%zu "
			"impulseBytes=%llu legacyImpulseBytes=%zu transformError=%.3g velocityError=%.3g\n",
			count, (unsigned long long)profile.contactPrepareDispatchCount,
			(unsigned long long)profile.contactPrepareDeviceRefreshCount, (unsigned long long)profile.contactSchedulePackCount,
			(unsigned long long)profile.contactScheduleReuseCount, (unsigned long long)profile.lastContactPrepareIndexBytes,
			(size_t)( ( count + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH ) * B3_SIMD_WIDTH * 144,
			(unsigned long long)profile.lastContactImpulseResultBytes,
			(size_t)( ( count + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH ) * sizeof( b3ContactConstraintWide ), maxTransformError,
			maxVelocityError );
	ENSURE( profile.contactPrepareDispatchCount == 4 );
	ENSURE( profile.contactPrepareFallbackCount == 0 );
	ENSURE( profile.contactPrepareDeviceRefreshCount == 3 * count );
	ENSURE( profile.contactSchedulePackCount == 1 );
	ENSURE( profile.contactScheduleReuseCount == 3 );
	ENSURE( profile.contactImpulseStoreBypassCount == 4 );
	ENSURE( profile.contactImpulseEventSyncCount == 0 );
	ENSURE( profile.contactImpulseSyncCount == 0 );
	ENSURE( profile.lastResidentConvexContactCount == count );
	ENSURE( profile.lastResidentConvexConstraintCount == ( count + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH );
	ENSURE( profile.lastContactPrepareIndexBytes ==
			(uint64_t)( ( count + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH ) * B3_SIMD_WIDTH * sizeof( uint32_t ) );
	ENSURE( profile.lastContactImpulseResultBytes == (uint64_t)count * sizeof( b3MetalContactImpulseResult ) );
	ENSURE( maxTransformError <= 3.0e-4f );
	ENSURE( maxVelocityError <= 3.0e-4f );

	// A graph insertion changes color-array topology/order and must force one
	// deterministic schedule rebuild before reuse can resume.
	b3BodyDef addedStaticDef = b3DefaultBodyDef();
	addedStaticDef.position = (b3Pos){ 0.0, 0.0, 3.0 * (double)count };
	b3BodyId cpuAddedStatic = b3CreateBody( cpuWorld, &addedStaticDef );
	b3BodyId gpuAddedStatic = b3CreateBody( gpuWorld, &addedStaticDef );
	b3CreateSphereShape( cpuAddedStatic, &staticShapeDef, &sphere );
	b3CreateSphereShape( gpuAddedStatic, &staticShapeDef, &sphere );
	b3BodyDef addedDynamicDef = b3DefaultBodyDef();
	addedDynamicDef.type = b3_dynamicBody;
	addedDynamicDef.enableSleep = false;
	addedDynamicDef.position = (b3Pos){ 0.96, 0.0, 3.0 * (double)count };
	b3BodyId cpuAddedDynamic = b3CreateBody( cpuWorld, &addedDynamicDef );
	b3BodyId gpuAddedDynamic = b3CreateBody( gpuWorld, &addedDynamicDef );
	b3CreateSphereShape( cpuAddedDynamic, &dynamicShapeDef, &sphere );
	b3CreateSphereShape( gpuAddedDynamic, &dynamicShapeDef, &sphere );
	b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	profile = b3World_GetMetalProfile( gpuWorld );
	ENSURE( profile.contactSchedulePackCount == 2 );
	ENSURE( profile.contactScheduleReuseCount == 3 );
	ENSURE( profile.contactImpulseStoreBypassCount == 5 );
	ENSURE( profile.lastResidentConvexContactCount == count + 1 );
	ENSURE( b3Length( b3Sub( b3Body_GetLinearVelocity( cpuAddedDynamic ), b3Body_GetLinearVelocity( gpuAddedDynamic ) ) ) <=
			3.0e-5f );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalResidentContactHitEventTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	worldDef.hitEventThreshold = 1.0f;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3World_SetContactRecycleDistance( cpuWorld, 0.0f );
	b3World_SetContactRecycleDistance( gpuWorld, 0.0f );
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BodyDef staticDef = b3DefaultBodyDef();
	b3BodyId cpuStatic = b3CreateBody( cpuWorld, &staticDef );
	b3BodyId gpuStatic = b3CreateBody( gpuWorld, &staticDef );
	b3CreateSphereShape( cpuStatic, &shapeDef, &sphere );
	b3CreateSphereShape( gpuStatic, &shapeDef, &sphere );
	b3BodyDef dynamicDef = b3DefaultBodyDef();
	dynamicDef.type = b3_dynamicBody;
	dynamicDef.enableSleep = false;
	dynamicDef.position = (b3Pos){ 0.99, 0.0, 0.0 };
	dynamicDef.linearVelocity = (b3Vec3){ -10.0f, 0.0f, 0.0f };
	b3BodyId cpuDynamic = b3CreateBody( cpuWorld, &dynamicDef );
	b3BodyId gpuDynamic = b3CreateBody( gpuWorld, &dynamicDef );
	shapeDef.enableHitEvents = true;
	b3CreateSphereShape( cpuDynamic, &shapeDef, &sphere );
	b3CreateSphereShape( gpuDynamic, &shapeDef, &sphere );

	b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	b3ContactEvents cpuEvents = b3World_GetContactEvents( cpuWorld );
	b3ContactEvents gpuEvents = b3World_GetContactEvents( gpuWorld );
	ENSURE( cpuEvents.hitCount == 1 );
	ENSURE( gpuEvents.hitCount == cpuEvents.hitCount );
	b3ContactHitEvent cpuHit = cpuEvents.hitEvents[0];
	b3ContactHitEvent gpuHit = gpuEvents.hitEvents[0];
	ENSURE( fabsf( cpuHit.approachSpeed - gpuHit.approachSpeed ) <= 1.0e-5f );
	ENSURE( b3Length( b3Sub( cpuHit.normal, gpuHit.normal ) ) <= 1.0e-5f );
	ENSURE( b3Length( b3SubPos( cpuHit.point, gpuHit.point ) ) <= 1.0e-5f );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	ENSURE( profile.contactPrepareDispatchCount == 1 );
	ENSURE( profile.lastContactImpulseResultBytes == sizeof( b3MetalContactImpulseResult ) );
	ENSURE( profile.contactImpulseStoreBypassCount == 1 );
	ENSURE( profile.contactImpulseEventSyncCount == 1 );
	ENSURE( profile.contactImpulseSyncCount == 1 );
	printf( "    resident hit event speed=%.3g impulseBytes=%llu cpuGpuMatch=yes\n", gpuHit.approachSpeed,
			(unsigned long long)profile.lastContactImpulseResultBytes );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalResidentWarmStartCarryTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3World_SetContactRecycleDistance( cpuWorld, 0.0f );
	b3World_SetContactRecycleDistance( gpuWorld, 0.0f );
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.5f };
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.6f;
	b3BodyDef staticDef = b3DefaultBodyDef();
	b3BodyId cpuStatic = b3CreateBody( cpuWorld, &staticDef );
	b3BodyId gpuStatic = b3CreateBody( gpuWorld, &staticDef );
	b3CreateSphereShape( cpuStatic, &shapeDef, &sphere );
	b3CreateSphereShape( gpuStatic, &shapeDef, &sphere );
	b3BodyDef dynamicDef = b3DefaultBodyDef();
	dynamicDef.type = b3_dynamicBody;
	dynamicDef.enableSleep = false;
	dynamicDef.position = (b3Pos){ 0.98, 0.0, 0.0 };
	dynamicDef.linearVelocity = (b3Vec3){ -0.4f, 0.1f, 0.0f };
	b3BodyId cpuDynamic = b3CreateBody( cpuWorld, &dynamicDef );
	b3BodyId gpuDynamic = b3CreateBody( gpuWorld, &dynamicDef );
	b3CreateSphereShape( cpuDynamic, &shapeDef, &sphere );
	b3ShapeId gpuDynamicShape = b3CreateSphereShape( gpuDynamic, &shapeDef, &sphere );

	b3World_Step( cpuWorld, 1.0f / 60.0f, 1 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 1 );
	b3World* gpu = b3GetWorldFromId( gpuWorld );
	int contactId = B3_NULL_INDEX;
	for ( int colorIndex = 0; colorIndex < B3_OVERFLOW_INDEX && contactId == B3_NULL_INDEX; ++colorIndex )
	{
		if ( gpu->constraintGraph.colors[colorIndex].convexContacts.count > 0 )
		{
			contactId = gpu->constraintGraph.colors[colorIndex].convexContacts.data[0];
		}
	}
	ENSURE( contactId != B3_NULL_INDEX );
	b3Contact* contact = b3Array_Get( gpu->contacts, contactId );
	ENSURE( contact->manifoldCount == 1 && contact->manifolds[0].pointCount == 1 );
	uint32_t resultGeneration = 0;
	int resultCount = 0;
	const b3MetalContactImpulseResult* results =
		b3MetalGetResidentContactImpulseTable( gpu->metalContext, &resultGeneration, &resultCount );
	ENSURE( results != NULL && contactId < resultCount );
	const b3MetalContactImpulseResult* result = results + contactId;
	ENSURE( result->generation == resultGeneration );
	ENSURE( result->contactGeneration == contact->generation );
	ENSURE( result->points[0].featureId == contact->manifolds[0].points[0].featureId );
	ENSURE( result->points[0].normalImpulse > 0.0f );
	contact->manifolds[0].points[0].normalImpulse = 999.0f;
	contact->manifolds[0].points[0].totalNormalImpulse = 999.0f;
	contact->manifolds[0].points[0].normalVelocity = 999.0f;
	b3ContactId publicContactId = { contactId + 1, gpu->worldId, 0, contact->generation };
	b3ContactData publicData = b3Contact_GetData( publicContactId );
	ENSURE( publicData.manifoldCount == 1 && publicData.manifolds == contact->manifolds );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == result->points[0].normalImpulse );
	ENSURE( contact->manifolds[0].points[0].totalNormalImpulse == result->points[0].totalNormalImpulse );
	ENSURE( contact->manifolds[0].points[0].normalVelocity == result->points[0].normalVelocity );
	contact->manifolds[0].points[0].normalImpulse = 998.0f;
	b3ContactData bodyData[1] = { 0 };
	ENSURE( b3Body_GetContactData( gpuDynamic, bodyData, 1 ) == 1 );
	ENSURE( bodyData[0].manifolds == contact->manifolds );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == result->points[0].normalImpulse );
	contact->manifolds[0].points[0].normalImpulse = 997.0f;
	b3ContactData shapeData[1] = { 0 };
	ENSURE( b3Shape_GetContactData( gpuDynamicShape, shapeData, 1 ) == 1 );
	ENSURE( shapeData[0].manifolds == contact->manifolds );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == result->points[0].normalImpulse );
	contact->manifolds[0].points[0].normalImpulse = 996.0f;
	b3Recording* recording = b3CreateRecording( 0 );
	ENSURE( recording != NULL );
	b3World_StartRecording( gpuWorld, recording );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == result->points[0].normalImpulse );
	b3World_StopRecording( gpuWorld );
	b3DestroyRecording( recording );
	uint32_t savedContactGeneration = contact->generation;
	contact->generation += 1;
	contact->manifolds[0].points[0].normalImpulse = 777.0f;
	ENSURE( b3MetalStageResidentContactPrepare( gpu->metalContext, contact ) );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == 777.0f );
	contact->generation = savedContactGeneration;
	ENSURE( b3MetalStageResidentContactPrepare( gpu->metalContext, contact ) );

	// Simulate a stale CPU public mirror. The next persistence pass must match
	// feature IDs against the resident result and recover every warm-start term.
	float residentNormalImpulse = result->points[0].normalImpulse;
	contact->manifolds[0].points[0].normalImpulse = 1000.0f;
	contact->manifolds[0].frictionImpulse = (b3Vec3){ 1000.0f, 1000.0f, 1000.0f };
	contact->manifolds[0].twistImpulse = 1000.0f;
	contact->manifolds[0].rollingImpulse = (b3Vec3){ 1000.0f, 1000.0f, 1000.0f };
	b3World_Step( cpuWorld, 1.0f / 60.0f, 1 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 1 );
	float velocityError = b3Length( b3Sub( b3Body_GetLinearVelocity( cpuDynamic ), b3Body_GetLinearVelocity( gpuDynamic ) ) );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	ENSURE( contact->manifolds[0].points[0].persisted );
	ENSURE( contact->manifolds[0].points[0].normalImpulse == residentNormalImpulse );
	ENSURE( profile.contactPrepareDispatchCount == 2 );
	ENSURE( profile.contactSchedulePackCount == 1 );
	ENSURE( profile.contactScheduleReuseCount == 1 );
	ENSURE( profile.contactPersistenceMatchCount >= 1 );
	ENSURE( profile.contactPrepareDeviceRefreshCount == 1 );
	ENSURE( profile.contactImpulseStoreBypassCount == 2 );
	ENSURE( profile.contactImpulseEventSyncCount == 0 );
	ENSURE( profile.contactImpulseSyncCount == 4 );
	ENSURE( velocityError <= 3.0e-5f );
	printf( "    resident warm-start feature=%u schedule=1/1 gpuPersistence=%llu deviceRefreshes=%llu velocityError=%.3g\n",
			result->points[0].featureId, (unsigned long long)profile.contactPersistenceMatchCount,
			(unsigned long long)profile.contactPrepareDeviceRefreshCount, velocityError );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalPairTraversalFallbackTest( void )
{
	const int bodyCount = 80;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	ENSURE( b3World_SetMetalBroadPhase( worldId, true ) );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 1.0f };
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
		b3CreateSphereShape( bodyId, &shapeDef, &sphere );
	}
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	printf( "    dense pair traversal dispatches=%llu fallbacks=%llu\n", (unsigned long long)profile.pairDispatchCount,
			(unsigned long long)profile.pairFallbackCount );
	ENSURE( profile.pairDispatchCount == 0 );
	ENSURE( profile.pairFallbackCount == 1 );
	ENSURE( b3World_SetMetalFinalization( worldId, true ) );
	b3World_DisableMetal( worldId );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.enabled == false );
	ENSURE( profile.finalizationEnabled == false );
	ENSURE( profile.broadPhaseEnabled == false );
	b3DestroyWorld( worldId );
	return 0;
}

static int MetalShapeCompactionTest( void )
{
	const int bodyCount = 1024;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	ENSURE( b3World_SetMetalFinalization( worldId, true ) );
	ENSURE( b3World_SetMetalBroadPhase( worldId, true ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	shapeDef.invokeContactCreation = false;
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.2f };
	int* movingProxyKeys = malloc( (size_t)( bodyCount / 2 ) * sizeof( int ) );
	b3ShapeId* movingShapeIds = malloc( (size_t)( bodyCount / 2 ) * sizeof( b3ShapeId ) );
	b3BodyId mutationBodyId = { 0 };
	ENSURE( movingProxyKeys != NULL && movingShapeIds != NULL );
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ 3.0f * (float)i, 0.0f, 0.0f };
		bodyDef.linearVelocity = i % 2 == 0 ? (b3Vec3){ 20.0f, 0.0f, 0.0f } : b3Vec3_zero;
		b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
		if ( i == 1 )
			mutationBodyId = bodyId;
		b3ShapeId shapeId = b3CreateSphereShape( bodyId, &shapeDef, &sphere );
		if ( i % 2 == 0 )
		{
			b3World* world = b3GetWorldFromId( worldId );
			movingProxyKeys[i / 2] = world->shapes.data[shapeId.index1 - 1].proxyKey;
			movingShapeIds[i / 2] = shapeId;
		}
	}

	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3World* world = b3GetWorldFromId( worldId );
	// Poison the CPU mirror without changing the broad-phase revision. The next
	// dispatch must use its prior resident fat bounds, not these packed values.
	for ( int i = 0; i < bodyCount / 2; ++i )
	{
		b3Shape* shape = world->shapes.data + movingShapeIds[i].index1 - 1;
		shape->fatAABB = (b3AABB){ { -1.0e6f, -1.0e6f, -1.0e6f }, { 1.0e6f, 1.0e6f, 1.0e6f } };
	}
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeCompactDispatchCount == 2 );
	ENSURE( profile.shapeBoundsResidentDispatchCount == 1 );
	ENSURE( profile.shapeInputPackCount == 1 );
	ENSURE( profile.shapeInputReuseCount == 1 );
	ENSURE( profile.lastShapeResultCount == bodyCount );
	ENSURE( profile.lastEnlargedShapeResultCount == bodyCount / 2 );
	ENSURE( world->broadPhase.moveArray.count == bodyCount / 2 );
	for ( int i = 0; i < bodyCount / 2; ++i )
	{
		ENSURE( world->broadPhase.moveArray.data[i] == movingProxyKeys[i] );
	}
	// Public AABB access materializes just the requested shape from the shared
	// Metal result while the rest of the CPU mirror remains deliberately stale.
	b3Shape* queryShape = world->shapes.data + movingShapeIds[0].index1 - 1;
	queryShape->aabb = (b3AABB){ { -1.0e6f, -1.0e6f, -1.0e6f }, { -9.0e5f, -9.0e5f, -9.0e5f } };
	b3AABB queryAABB = b3Shape_GetAABB( movingShapeIds[0] );
	ENSURE( queryAABB.lowerBound.x > -1000.0f );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeResultApplyCount == 0 );
	ENSURE( profile.shapeBoundsSyncCount == 1 );
	// A public transform synchronizes stale peers and invalidates the packed
	// registry even when the new AABB remains inside the existing fat bound.
	b3Body_SetTransform( mutationBodyId, (b3Pos){ 3.0f, 0.01f, 0.0f }, b3Quat_identity );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeBoundsResidentDispatchCount == 1 );
	ENSURE( profile.shapeInputPackCount == 2 );
	ENSURE( profile.shapeInputReuseCount == 1 );
	ENSURE( profile.shapeResultApplyCount == 0 );
	ENSURE( profile.shapeBoundsSyncCount == bodyCount );
	printf( "    shape compaction enlarged=%d/%d stableOrder=yes residentBounds=%llu inputPacks=%llu inputReuses=%llu "
			"fullApplies=%llu syncShapes=%llu\n",
			profile.lastEnlargedShapeResultCount, profile.lastShapeResultCount,
			(unsigned long long)profile.shapeBoundsResidentDispatchCount, (unsigned long long)profile.shapeInputPackCount,
			(unsigned long long)profile.shapeInputReuseCount, (unsigned long long)profile.shapeResultApplyCount,
			(unsigned long long)profile.shapeBoundsSyncCount );

	b3World_DisableMetal( worldId );
	ENSURE( world->metalShapeCpuBoundsStale == false );
	ENSURE( world->metalShapeBoundsSyncCount == 2 * bodyCount );
	free( movingShapeIds );
	free( movingProxyKeys );
	b3DestroyWorld( worldId );
	return 0;
}

static int MetalShapeInputRegistryTest( void )
{
	const int bodyCount = 64;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableContinuous = false;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( worldId, 1 ) );
	ENSURE( b3World_SetMetalFinalization( worldId, true ) );
	ENSURE( b3World_SetMetalBroadPhase( worldId, true ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	shapeDef.invokeContactCreation = false;
	b3Sphere sphere = { .center = b3Vec3_zero, .radius = 0.2f };
	b3BodyId bodies[bodyCount];
	b3ShapeId shapes[bodyCount];
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 2.0f * (float)i, 0.0f, 0.0f };
		bodies[i] = b3CreateBody( worldId, &bodyDef );
		shapes[i] = b3CreateSphereShape( bodies[i], &shapeDef, &sphere );
	}

	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3MetalProfile profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeInputPackCount == 1 );
	ENSURE( profile.shapeInputReuseCount == 1 );

	// Sleeping a middle body swap-removes an awake sim without changing shape
	// geometry. Exact cached body ids must reject the old bodyIndex mapping.
	b3Body_SetAwake( bodies[17], false );
	ENSURE( b3Body_IsAwake( bodies[17] ) == false );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeInputPackCount == 2 );
	ENSURE( profile.shapeInputReuseCount == 1 );
	ENSURE( profile.shapeBoundsSyncCount == bodyCount );

	b3Body_SetAwake( bodies[17], true );
	ENSURE( b3Body_IsAwake( bodies[17] ) );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	printf( "    shape registry packs=%llu reuses=%llu awakeReorders=2 syncShapes=%llu\n",
			(unsigned long long)profile.shapeInputPackCount, (unsigned long long)profile.shapeInputReuseCount,
			(unsigned long long)profile.shapeBoundsSyncCount );
	ENSURE( profile.shapeInputPackCount == 3 );
	ENSURE( profile.shapeInputReuseCount == 2 );
	ENSURE( profile.shapeBoundsSyncCount == 2 * bodyCount - 1 );

	// Filter-only edits may deliberately skip immediate contact invocation and
	// therefore do not need to mutate tree topology. They still invalidate the
	// cached eligibility bit and force an exact registry rebuild.
	b3Filter filter = b3Shape_GetFilter( shapes[9] );
	filter.maskBits = 1;
	b3Shape_SetFilter( shapes[9], filter, false );
	b3World_Step( worldId, 1.0f / 60.0f, 1 );
	profile = b3World_GetMetalProfile( worldId );
	ENSURE( profile.shapeInputPackCount == 4 );
	ENSURE( profile.shapeInputReuseCount == 2 );
	ENSURE( profile.shapeBoundsSyncCount == 3 * bodyCount - 1 );
	ENSURE( profile.shapeResultApplyCount == 1 );

	b3DestroyWorld( worldId );
	return 0;
}

static int MetalWorldIntegrationTest( void )
{
	const int count = 2048;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	worldDef.enableContinuous = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	ENSURE( b3World_SetMetalFinalization( gpuWorld, true ) );
	ENSURE( b3World_SetMetalBroadPhase( gpuWorld, true ) );

	b3BodyId* cpuBodies = malloc( (size_t)count * sizeof( b3BodyId ) );
	b3BodyId* gpuBodies = malloc( (size_t)count * sizeof( b3BodyId ) );
	b3ShapeId* cpuShapes = malloc( (size_t)count * sizeof( b3ShapeId ) );
	b3ShapeId* gpuShapes = malloc( (size_t)count * sizeof( b3ShapeId ) );
	ENSURE( cpuBodies != NULL && gpuBodies != NULL && cpuShapes != NULL && gpuShapes != NULL );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	shapeDef.invokeContactCreation = false;
	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
#if defined( BOX3D_DOUBLE_PRECISION )
		bodyDef.position =
			(b3Pos){ 1.0e8 + 32.0 * (double)( i % 64 ), -1.0e8 + 32.0 * (double)( i / 64 ), 0.1 * (double)( i % 7 ) };
#else
		bodyDef.position = (b3Pos){ (float)( i % 64 ), (float)( i / 64 ), 0.1f * (float)( i % 7 ) };
#endif
		bodyDef.linearVelocity = (b3Vec3){ b3MetalRandomFloat( -40.0f, 40.0f ), b3MetalRandomFloat( -40.0f, 40.0f ),
										   b3MetalRandomFloat( -40.0f, 40.0f ) };
		bodyDef.angularVelocity = (b3Vec3){ b3MetalRandomFloat( -20.0f, 20.0f ), b3MetalRandomFloat( -20.0f, 20.0f ),
											b3MetalRandomFloat( -20.0f, 20.0f ) };
		cpuBodies[i] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &bodyDef );
		if ( i % 3 == 0 )
		{
			b3Sphere sphere = { .center = { 0.13f, -0.09f, 0.07f }, .radius = 0.21f };
			cpuShapes[i] = b3CreateSphereShape( cpuBodies[i], &shapeDef, &sphere );
			gpuShapes[i] = b3CreateSphereShape( gpuBodies[i], &shapeDef, &sphere );
		}
		else if ( i % 3 == 1 )
		{
			b3Capsule capsule = { .center1 = { -0.18f, -0.12f, 0.05f }, .center2 = { 0.16f, 0.22f, -0.08f }, .radius = 0.11f };
			cpuShapes[i] = b3CreateCapsuleShape( cpuBodies[i], &shapeDef, &capsule );
			gpuShapes[i] = b3CreateCapsuleShape( gpuBodies[i], &shapeDef, &capsule );
		}
		else
		{
			b3BoxHull hull = b3MakeOffsetBoxHull( 0.17f, 0.23f, 0.14f, (b3Vec3){ 0.12f, -0.08f, 0.09f } );
			cpuShapes[i] = b3CreateHullShape( cpuBodies[i], &shapeDef, &hull.base );
			gpuShapes[i] = b3CreateHullShape( gpuBodies[i], &shapeDef, &hull.base );
		}
	}

	b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	b3World* gpuWorldInternal = b3GetWorldFromId( gpuWorld );
	ENSURE( VerifyResidentPairTraversal( gpuWorldInternal ) == 0 );
	ENSURE( gpuWorldInternal->metalShapeCpuBoundsStale );
	b3AABB firstBodyAABB = b3Body_ComputeAABB( gpuBodies[0] );
	ENSURE( b3IsValidAABB( firstBodyAABB ) );

	float maxPositionError = 0.0f;
	float maxRotationError = 0.0f;
	float maxAABBError = 0.0f;
#if defined( BOX3D_DOUBLE_PRECISION )
	float maxOracleUnderflow = 0.0f;
#endif
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxPositionError = b3MaxFloat( maxPositionError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxPositionError = b3MaxFloat( maxPositionError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxPositionError = b3MaxFloat( maxPositionError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxRotationError = b3MaxFloat( maxRotationError, fabsf( a.q.v.x - b.q.v.x ) );
		maxRotationError = b3MaxFloat( maxRotationError, fabsf( a.q.v.y - b.q.v.y ) );
		maxRotationError = b3MaxFloat( maxRotationError, fabsf( a.q.v.z - b.q.v.z ) );
		maxRotationError = b3MaxFloat( maxRotationError, fabsf( a.q.s - b.q.s ) );
		b3AABB cpuAABB = b3Shape_GetAABB( cpuShapes[i] );
		b3AABB gpuAABB = b3Shape_GetAABB( gpuShapes[i] );
#if defined( BOX3D_DOUBLE_PRECISION )
		b3Shape* gpuShape = b3Array_Get( gpuWorldInternal->shapes, gpuShapes[i].index1 - 1 );
		b3Body* gpuBody = b3Array_Get( gpuWorldInternal->bodies, gpuShape->bodyId );
		b3AABB oracle =
			b3ComputeFatShapeAABB( gpuShape, b3GetBodyTransformQuick( gpuWorldInternal, gpuBody ), B3_SPECULATIVE_DISTANCE );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, gpuAABB.lowerBound.x - oracle.lowerBound.x );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, gpuAABB.lowerBound.y - oracle.lowerBound.y );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, gpuAABB.lowerBound.z - oracle.lowerBound.z );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, oracle.upperBound.x - gpuAABB.upperBound.x );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, oracle.upperBound.y - gpuAABB.upperBound.y );
		maxOracleUnderflow = b3MaxFloat( maxOracleUnderflow, oracle.upperBound.z - gpuAABB.upperBound.z );
#endif
		const float* cpuBounds = (const float*)&cpuAABB;
		const float* gpuBounds = (const float*)&gpuAABB;
		for ( int j = 0; j < 6; ++j )
		{
			maxAABBError = b3MaxFloat( maxAABBError, fabsf( cpuBounds[j] - gpuBounds[j] ) );
		}
	}

	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    integrated world device=%s fusedDispatches=%llu shapeDispatches=%llu compact=%d/%d "
			"fullApplies=%llu syncShapes=%llu maxPositionError=%.3g maxRotationError=%.3g maxAABBError=%.3g\n",
			profile.deviceName, (unsigned long long)profile.unconstrainedDispatchCount,
			(unsigned long long)profile.shapeDispatchCount, profile.lastEnlargedShapeResultCount, profile.lastShapeResultCount,
			(unsigned long long)profile.shapeResultApplyCount, (unsigned long long)profile.shapeBoundsSyncCount, maxPositionError,
			maxRotationError, maxAABBError );
	ENSURE( profile.enabled );
	ENSURE( profile.unconstrainedDispatchCount == 4 );
	ENSURE( profile.unconstrainedFallbackCount == 0 );
	ENSURE( profile.finalizationEnabled );
	ENSURE( profile.finalizationDispatchCount == 1 );
	ENSURE( profile.shapeDispatchCount == 1 );
	ENSURE( profile.shapeFallbackCount == 0 );
	ENSURE( profile.shapeCompactDispatchCount == 1 );
	ENSURE( profile.shapeResultApplyCount == 0 );
	ENSURE( profile.shapeBoundsSyncCount == count );
	ENSURE( profile.lastShapeResultCount == count );
	ENSURE( 0 < profile.lastEnlargedShapeResultCount && profile.lastEnlargedShapeResultCount <= count );
	ENSURE( gpuWorldInternal->broadPhase.moveArray.count == profile.lastEnlargedShapeResultCount );
	ENSURE( profile.positionDispatchCount == 0 );
	ENSURE( profile.positionFallbackCount == 0 );
	ENSURE( maxPositionError <= 3.0e-5f );
	ENSURE( maxRotationError <= 3.0e-5f );
	ENSURE( maxAABBError <= 5.0e-5f );
#if defined( BOX3D_DOUBLE_PRECISION )
	printf( "    VF64 far-world oracle underflow=%.9g\n", maxOracleUnderflow );
	ENSURE( maxOracleUnderflow <= 0.0f );
#endif

	free( gpuShapes );
	free( cpuShapes );
	free( gpuBodies );
	free( cpuBodies );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalDistanceJointContactTest( void )
{
	const int count = 16;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BoxHull groundHull = b3MakeBoxHull( 20.0f, 1.0f, 4.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId cpuGround = b3CreateBody( cpuWorld, &groundDef );
	b3BodyId gpuGround = b3CreateBody( gpuWorld, &groundDef );
	b3CreateHullShape( cpuGround, &shapeDef, &groundHull.base );
	b3CreateHullShape( gpuGround, &shapeDef, &groundHull.base );

	b3BoxHull boxHull = b3MakeBoxHull( 0.45f, 0.5f, 0.45f );
	b3BodyId cpuBodies[count];
	b3BodyId gpuBodies[count];
	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -7.5f + (float)i, 0.48f, 0.0f };
		cpuBodies[i] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateHullShape( cpuBodies[i], &shapeDef, &boxHull.base );
		b3CreateHullShape( gpuBodies[i], &shapeDef, &boxHull.base );
	}
	b3DistanceJointDef cpuJointDef = b3DefaultDistanceJointDef();
	cpuJointDef.base.bodyIdA = cpuBodies[0];
	cpuJointDef.base.bodyIdB = cpuBodies[1];
	cpuJointDef.length = 1.0f;
	b3CreateDistanceJoint( cpuWorld, &cpuJointDef );
	b3DistanceJointDef gpuJointDef = cpuJointDef;
	gpuJointDef.base.bodyIdA = gpuBodies[0];
	gpuJointDef.base.bodyIdB = gpuBodies[1];
	b3CreateDistanceJoint( gpuWorld, &gpuJointDef );

	b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	float maxError = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.x - b.q.v.x ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.y - b.q.v.y ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.z - b.q.v.z ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.s - b.q.s ) );
	}
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    distance joint+contacts jointDispatches=%llu contactDispatches=%llu maxError=%.3g\n",
			(unsigned long long)profile.jointDispatchCount, (unsigned long long)profile.contactDispatchCount, maxError );
	ENSURE( profile.unconstrainedDispatchCount == 0 );
	ENSURE( profile.unconstrainedFallbackCount == 0 );
	ENSURE( profile.contactDispatchCount == 4 );
	ENSURE( profile.contactFallbackCount == 0 );
	ENSURE( profile.jointDispatchCount == 4 );
	ENSURE( profile.jointFallbackCount == 0 );
	ENSURE( profile.positionDispatchCount == 0 );
	ENSURE( maxError <= 3.0e-5f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalUnsupportedJointFallbackTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3BodyDef bodyDefA = b3DefaultBodyDef();
	bodyDefA.type = b3_dynamicBody;
	bodyDefA.enableSleep = false;
	bodyDefA.position = (b3Pos){ -0.5f, 0.0f, 0.0f };
	b3BodyDef bodyDefB = bodyDefA;
	bodyDefB.position = (b3Pos){ 0.5f, 0.0f, 0.0f };
	bodyDefB.linearVelocity = (b3Vec3){ 0.0f, 1.0f, 0.0f };
	b3BodyId cpuA = b3CreateBody( cpuWorld, &bodyDefA );
	b3BodyId cpuB = b3CreateBody( cpuWorld, &bodyDefB );
	b3BodyId gpuA = b3CreateBody( gpuWorld, &bodyDefA );
	b3BodyId gpuB = b3CreateBody( gpuWorld, &bodyDefB );

	b3RevoluteJointDef cpuDef = b3DefaultRevoluteJointDef();
	cpuDef.base.bodyIdA = cpuA;
	cpuDef.base.bodyIdB = cpuB;
	cpuDef.base.localFrameA.p = (b3Vec3){ 0.5f, 0.0f, 0.0f };
	cpuDef.base.localFrameB.p = (b3Vec3){ -0.5f, 0.0f, 0.0f };
	b3CreateRevoluteJoint( cpuWorld, &cpuDef );
	b3RevoluteJointDef gpuDef = cpuDef;
	gpuDef.base.bodyIdA = gpuA;
	gpuDef.base.bodyIdB = gpuB;
	b3CreateRevoluteJoint( gpuWorld, &gpuDef );

	b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	b3WorldTransform cpuTransform = b3Body_GetTransform( cpuB );
	b3WorldTransform gpuTransform = b3Body_GetTransform( gpuB );
	float maxError = b3MaxFloat( (float)fabs( (double)cpuTransform.p.x - (double)gpuTransform.p.x ),
								 (float)fabs( (double)cpuTransform.p.y - (double)gpuTransform.p.y ) );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    unsupported revolute jointFallbacks=%llu positionDispatches=%llu maxError=%.3g\n",
			(unsigned long long)profile.jointFallbackCount, (unsigned long long)profile.positionDispatchCount, maxError );
	ENSURE( profile.jointDispatchCount == 0 );
	ENSURE( profile.jointFallbackCount == 4 );
	ENSURE( profile.positionDispatchCount == 4 );
	ENSURE( maxError <= 3.0e-5f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalConvexFrictionContactTest( void )
{
	const int count = 128;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	ENSURE( b3World_SetMetalFinalization( gpuWorld, true ) );
	ENSURE( b3World_SetMetalBroadPhase( gpuWorld, true ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.65f;
	shapeDef.baseMaterial.restitution = 0.0f;
	shapeDef.baseMaterial.rollingResistance = 0.1f;
	b3ShapeDef groundShapeDef = shapeDef;
	groundShapeDef.baseMaterial.tangentVelocity = (b3Vec3){ 0.3f, 0.0f, 0.15f };
	b3BoxHull groundHull = b3MakeBoxHull( 80.0f, 1.0f, 4.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId cpuGround = b3CreateBody( cpuWorld, &groundDef );
	b3BodyId gpuGround = b3CreateBody( gpuWorld, &groundDef );
	b3CreateHullShape( cpuGround, &groundShapeDef, &groundHull.base );
	b3CreateHullShape( gpuGround, &groundShapeDef, &groundHull.base );

	b3BoxHull boxHull = b3MakeBoxHull( 0.4f, 0.5f, 0.4f );
	b3BodyId cpuBodies[count];
	b3BodyId gpuBodies[count];
	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -7.5f + (float)( i % 16 ), 0.47f + 0.98f * (float)( i / 16 ), 0.0f };
		bodyDef.linearVelocity = (b3Vec3){ 0.01f * (float)( i % 5 - 2 ), -0.25f - 0.001f * (float)i, 0.0f };
		bodyDef.angularVelocity = (b3Vec3){ 0.0f, 0.0f, 0.02f * (float)( i % 3 - 1 ) };
		cpuBodies[i] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateHullShape( cpuBodies[i], &shapeDef, &boxHull.base );
		b3CreateHullShape( gpuBodies[i], &shapeDef, &boxHull.base );
	}

	for ( int step = 0; step < 10; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}

	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    convex friction contacts dispatches=%llu treeUploads=%llu treeRefits=%llu gpu=%.3f ms "
			"transformError=%.3g velocityError=%.3g\n",
			(unsigned long long)profile.contactDispatchCount, (unsigned long long)profile.pairTreeUploadCount,
			(unsigned long long)profile.pairTreeRefitCount, profile.lastContactGpuMilliseconds, maxTransformError,
			maxVelocityError );
	ENSURE( profile.contactDispatchCount == 40 );
	ENSURE( profile.pairDispatchCount >= 1 );
	ENSURE( profile.pairFallbackCount == 0 );
	ENSURE( profile.finalizationDispatchCount == 10 );
	ENSURE( profile.shapeDispatchCount == 10 );
	ENSURE( profile.shapeFallbackCount == 0 );
	ENSURE( profile.shapeInputPackCount == 1 );
	ENSURE( profile.shapeInputReuseCount == 9 );
	ENSURE( profile.shapeResultApplyCount == 10 );
	ENSURE( profile.shapeBoundsSyncCount == 0 );
	ENSURE( profile.pairTreeUploadCount == 1 );
	ENSURE( profile.pairTreeRefitCount == 10 );
	ENSURE( profile.positionDispatchCount == 0 );
	ENSURE( profile.unconstrainedDispatchCount == 0 );
	ENSURE( maxTransformError <= 2.0e-4f );
	ENSURE( maxVelocityError <= 2.0e-4f );

	// Disabling the resident broad-phase must restore the CPU tree invariant.
	ENSURE( b3World_SetMetalBroadPhase( gpuWorld, false ) );
	b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalConvexRestitutionContactTest( void )
{
	const int count = 64;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.4f;
	shapeDef.baseMaterial.restitution = 0.65f;
	shapeDef.baseMaterial.rollingResistance = 0.15f;
	b3BoxHull groundHull = b3MakeBoxHull( 45.0f, 1.0f, 3.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
	b3BodyId cpuGround = b3CreateBody( cpuWorld, &groundDef );
	b3BodyId gpuGround = b3CreateBody( gpuWorld, &groundDef );
	b3CreateHullShape( cpuGround, &shapeDef, &groundHull.base );
	b3CreateHullShape( gpuGround, &shapeDef, &groundHull.base );

	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
	b3BodyId cpuBodies[count];
	b3BodyId gpuBodies[count];
	for ( int i = 0; i < count; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -37.8f + 1.2f * (float)i, 0.48f, 0.0f };
		bodyDef.linearVelocity = (b3Vec3){ 0.03f * (float)( i % 5 - 2 ), -4.0f - 0.01f * (float)i, 0.02f * (float)( i % 3 - 1 ) };
		bodyDef.angularVelocity = (b3Vec3){ 0.2f, 0.1f * (float)( i % 4 ), -0.15f };
		cpuBodies[i] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateSphereShape( cpuBodies[i], &shapeDef, &sphere );
		b3CreateSphereShape( gpuBodies[i], &shapeDef, &sphere );
	}

	for ( int step = 0; step < 6; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	float maxCpuUpwardVelocity = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		maxCpuUpwardVelocity = b3MaxFloat( maxCpuUpwardVelocity, av.y );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}

	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    convex restitution dispatches=%llu upward=%.3g transformError=%.3g velocityError=%.3g\n",
			(unsigned long long)profile.contactDispatchCount, maxCpuUpwardVelocity, maxTransformError, maxVelocityError );
	ENSURE( profile.contactDispatchCount >= 4 );
	ENSURE( maxCpuUpwardVelocity > 1.0f );
	ENSURE( maxTransformError <= 2.0e-4f );
	ENSURE( maxVelocityError <= 2.0e-4f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalMeshContactTest( void )
{
	const int count = 64;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3Vec3 vertices[4] = {
		{ -20.0f, 0.0f, -20.0f },
		{ 20.0f, 0.0f, -20.0f },
		{ 20.0f, 0.0f, 20.0f },
		{ -20.0f, 0.0f, 20.0f },
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
	ENSURE( mesh != NULL );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.55f;
	shapeDef.baseMaterial.restitution = 0.15f;
	shapeDef.baseMaterial.rollingResistance = 0.04f;
	b3BodyDef groundDef = b3DefaultBodyDef();
	b3BodyId cpuGround = b3CreateBody( cpuWorld, &groundDef );
	b3BodyId gpuGround = b3CreateBody( gpuWorld, &groundDef );
	b3CreateMeshShape( cpuGround, &shapeDef, mesh, (b3Vec3){ 1.0f, 1.0f, 1.0f } );
	b3CreateMeshShape( gpuGround, &shapeDef, mesh, (b3Vec3){ 1.0f, 1.0f, 1.0f } );

	b3BoxHull boxHull = b3MakeBoxHull( 0.45f, 0.5f, 0.45f );
	b3BodyId cpuBodies[count];
	b3BodyId gpuBodies[count];
	for ( int i = 0; i < count; ++i )
	{
		int baseIndex = i % 32;
		int layer = i / 32;
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ -7.0f + 2.0f * (float)( baseIndex % 8 ), 0.48f + 0.98f * (float)layer,
									-3.0f + 2.0f * (float)( baseIndex / 8 ) };
		bodyDef.linearVelocity = (b3Vec3){ 0.02f * (float)( i % 5 - 2 ), -0.5f, 0.015f * (float)( i % 7 - 3 ) };
		bodyDef.angularVelocity = (b3Vec3){ 0.03f, 0.02f * (float)( i % 3 ), -0.04f };
		cpuBodies[i] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateHullShape( cpuBodies[i], &shapeDef, &boxHull.base );
		b3CreateHullShape( gpuBodies[i], &shapeDef, &boxHull.base );
	}

	for ( int step = 0; step < 10; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    mesh contacts dispatches=%llu transformError=%.3g velocityError=%.3g\n",
			(unsigned long long)profile.contactDispatchCount, maxTransformError, maxVelocityError );
	ENSURE( profile.contactDispatchCount == 40 );
	ENSURE( profile.positionDispatchCount == 0 );
	ENSURE( maxTransformError <= 3.0e-4f );
	ENSURE( maxVelocityError <= 3.0e-4f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	b3DestroyMesh( mesh );
	return 0;
}

static int MetalOverflowContactTest( void )
{
	const int spokeCount = 32;
	const int bodyCount = spokeCount + 1;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.baseMaterial.friction = 0.37f;
	shapeDef.baseMaterial.restitution = 0.21f;
	shapeDef.baseMaterial.rollingResistance = 0.03f;
	b3Sphere hubSphere = { { 0.0f, 0.0f, 0.0f }, 3.0f };
	b3Sphere spokeSphere = { { 0.0f, 0.0f, 0.0f }, 0.25f };
	b3BodyId cpuBodies[bodyCount];
	b3BodyId gpuBodies[bodyCount];

	b3BodyDef hubDef = b3DefaultBodyDef();
	hubDef.type = b3_dynamicBody;
	hubDef.enableSleep = false;
	hubDef.angularVelocity = (b3Vec3){ 0.07f, -0.04f, 0.03f };
	cpuBodies[0] = b3CreateBody( cpuWorld, &hubDef );
	gpuBodies[0] = b3CreateBody( gpuWorld, &hubDef );
	b3CreateSphereShape( cpuBodies[0], &shapeDef, &hubSphere );
	b3CreateSphereShape( gpuBodies[0], &shapeDef, &hubSphere );

	for ( int i = 0; i < spokeCount; ++i )
	{
		float angle = 2.0f * B3_PI * (float)i / (float)spokeCount;
		float x = 3.20f * cosf( angle );
		float z = 3.20f * sinf( angle );
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ x, 0.0f, z };
		bodyDef.linearVelocity = (b3Vec3){ -0.2f * cosf( angle ), 0.01f * (float)( i % 3 - 1 ), -0.2f * sinf( angle ) };
		bodyDef.angularVelocity = (b3Vec3){ 0.01f * (float)( i % 5 ), -0.02f, 0.015f };
		cpuBodies[i + 1] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i + 1] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateSphereShape( cpuBodies[i + 1], &shapeDef, &spokeSphere );
		b3CreateSphereShape( gpuBodies[i + 1], &shapeDef, &spokeSphere );
	}

	for ( int step = 0; step < 3; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}

	b3Counters counters = b3World_GetCounters( cpuWorld );
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}

	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    overflow contacts=%d dispatches=%llu fallbacks=%llu transformError=%.3g velocityError=%.3g\n",
			counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1], (unsigned long long)profile.contactDispatchCount,
			(unsigned long long)profile.contactFallbackCount, maxTransformError, maxVelocityError );
	ENSURE( counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1] > 0 );
	ENSURE( profile.contactDispatchCount == 12 );
	ENSURE( profile.contactFallbackCount == 0 );
	ENSURE( profile.positionDispatchCount == 0 );
	ENSURE( maxTransformError <= 5.0e-4f );
	ENSURE( maxVelocityError <= 5.0e-4f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalDistanceJointModesAndOverflowTest( void )
{
	const int spokeCount = 32;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -1.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.2f };
	b3BodyId cpuBodies[spokeCount + 1];
	b3BodyId gpuBodies[spokeCount + 1];
	b3BodyDef hubDef = b3DefaultBodyDef();
	hubDef.type = b3_dynamicBody;
	hubDef.enableSleep = false;
	hubDef.angularVelocity = (b3Vec3){ 0.03f, -0.02f, 0.04f };
	cpuBodies[0] = b3CreateBody( cpuWorld, &hubDef );
	gpuBodies[0] = b3CreateBody( gpuWorld, &hubDef );
	b3CreateSphereShape( cpuBodies[0], &shapeDef, &sphere );
	b3CreateSphereShape( gpuBodies[0], &shapeDef, &sphere );

	for ( int i = 0; i < spokeCount; ++i )
	{
		float angle = 2.0f * B3_PI * (float)i / (float)spokeCount;
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ 2.0f * cosf( angle ), 0.05f * (float)( i % 3 - 1 ), 2.0f * sinf( angle ) };
		bodyDef.linearVelocity =
			(b3Vec3){ 0.04f * (float)( i % 5 - 2 ), 0.03f * (float)( i % 4 - 1 ), -0.025f * (float)( i % 7 - 3 ) };
		cpuBodies[i + 1] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i + 1] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateSphereShape( cpuBodies[i + 1], &shapeDef, &sphere );
		b3CreateSphereShape( gpuBodies[i + 1], &shapeDef, &sphere );

		b3DistanceJointDef cpuDef = b3DefaultDistanceJointDef();
		cpuDef.base.bodyIdA = cpuBodies[0];
		cpuDef.base.bodyIdB = cpuBodies[i + 1];
		cpuDef.length = 1.85f + 0.01f * (float)( i % 4 );
		if ( i % 4 != 0 )
		{
			cpuDef.enableSpring = true;
			cpuDef.hertz = i % 4 == 1 ? 3.0f : 0.0f;
			cpuDef.dampingRatio = 0.65f;
			cpuDef.lowerSpringForce = -40.0f;
			cpuDef.upperSpringForce = 55.0f;
		}
		if ( i % 4 >= 2 )
		{
			cpuDef.enableLimit = true;
			cpuDef.minLength = 1.75f;
			cpuDef.maxLength = 1.95f;
		}
		if ( i % 4 == 3 )
		{
			cpuDef.enableMotor = true;
			cpuDef.maxMotorForce = 12.0f;
			cpuDef.motorSpeed = ( i & 1 ) ? 0.2f : -0.15f;
		}
		b3CreateDistanceJoint( cpuWorld, &cpuDef );
		b3DistanceJointDef gpuDef = cpuDef;
		gpuDef.base.bodyIdA = gpuBodies[0];
		gpuDef.base.bodyIdB = gpuBodies[i + 1];
		b3CreateDistanceJoint( gpuWorld, &gpuDef );
	}

	for ( int step = 0; step < 6; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}

	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i <= spokeCount; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}
	b3Counters counters = b3World_GetCounters( cpuWorld );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    distance modes overflow=%d dispatches=%llu transformError=%.3g velocityError=%.3g\n",
			counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1], (unsigned long long)profile.jointDispatchCount, maxTransformError,
			maxVelocityError );
	ENSURE( counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1] > 0 );
	ENSURE( profile.jointDispatchCount == 24 );
	ENSURE( profile.jointFallbackCount == 0 );
	ENSURE( profile.contactDispatchCount == 0 );
	ENSURE( maxTransformError <= 7.0e-4f );
	ENSURE( maxVelocityError <= 7.0e-4f );

	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalMixedDistanceParallelJointTest( void )
{
	const int pairCount = 32;
	const int bodyCount = 2 * pairCount;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -2.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3BoxHull hull = b3MakeBoxHull( 0.2f, 0.3f, 0.25f );
	b3BodyId cpuBodies[bodyCount];
	b3BodyId gpuBodies[bodyCount];
	for ( int pair = 0; pair < pairCount; ++pair )
	{
		b3BodyDef aDef = b3DefaultBodyDef();
		aDef.type = b3_dynamicBody;
		aDef.enableSleep = false;
		aDef.position = (b3Pos){ 3.0f * (float)( pair % 8 ), 2.0f * (float)( pair / 8 ), 0.0f };
		aDef.angularVelocity = (b3Vec3){ 0.02f, -0.03f, 0.01f };
		b3BodyDef bDef = aDef;
		bDef.position.x += 1.25f;
		bDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 1.0f, 0.5f, 0.2f } ), 0.08f + 0.003f * (float)pair );
		bDef.angularVelocity = (b3Vec3){ -0.08f, 0.04f, -0.02f };
		cpuBodies[2 * pair] = b3CreateBody( cpuWorld, &aDef );
		cpuBodies[2 * pair + 1] = b3CreateBody( cpuWorld, &bDef );
		gpuBodies[2 * pair] = b3CreateBody( gpuWorld, &aDef );
		gpuBodies[2 * pair + 1] = b3CreateBody( gpuWorld, &bDef );
		b3CreateHullShape( cpuBodies[2 * pair], &shapeDef, &hull.base );
		b3CreateHullShape( cpuBodies[2 * pair + 1], &shapeDef, &hull.base );
		b3CreateHullShape( gpuBodies[2 * pair], &shapeDef, &hull.base );
		b3CreateHullShape( gpuBodies[2 * pair + 1], &shapeDef, &hull.base );
		if ( pair & 1 )
		{
			b3ParallelJointDef cpuDef = b3DefaultParallelJointDef();
			cpuDef.base.bodyIdA = cpuBodies[2 * pair];
			cpuDef.base.bodyIdB = cpuBodies[2 * pair + 1];
			cpuDef.hertz = 2.0f + 0.1f * (float)( pair % 5 );
			cpuDef.dampingRatio = 0.55f;
			cpuDef.maxTorque = 20.0f + (float)pair;
			b3CreateParallelJoint( cpuWorld, &cpuDef );
			b3ParallelJointDef gpuDef = cpuDef;
			gpuDef.base.bodyIdA = gpuBodies[2 * pair];
			gpuDef.base.bodyIdB = gpuBodies[2 * pair + 1];
			b3CreateParallelJoint( gpuWorld, &gpuDef );
		}
		else
		{
			b3DistanceJointDef cpuDef = b3DefaultDistanceJointDef();
			cpuDef.base.bodyIdA = cpuBodies[2 * pair];
			cpuDef.base.bodyIdB = cpuBodies[2 * pair + 1];
			cpuDef.length = 1.0f;
			b3CreateDistanceJoint( cpuWorld, &cpuDef );
			b3DistanceJointDef gpuDef = cpuDef;
			gpuDef.base.bodyIdA = gpuBodies[2 * pair];
			gpuDef.base.bodyIdB = gpuBodies[2 * pair + 1];
			b3CreateDistanceJoint( gpuWorld, &gpuDef );
		}
	}

	for ( int step = 0; step < 10; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i < bodyCount; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		b3Vec3 av = b3Body_GetLinearVelocity( cpuBodies[i] );
		b3Vec3 bv = b3Body_GetLinearVelocity( gpuBodies[i] );
		b3Vec3 aw = b3Body_GetAngularVelocity( cpuBodies[i] );
		b3Vec3 bw = b3Body_GetAngularVelocity( gpuBodies[i] );
		for ( int j = 0; j < 3; ++j )
		{
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&av )[j] - ( (float*)&bv )[j] ) );
			maxVelocityError = b3MaxFloat( maxVelocityError, fabsf( ( (float*)&aw )[j] - ( (float*)&bw )[j] ) );
		}
	}
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    mixed distance+parallel dispatches=%llu transformError=%.3g velocityError=%.3g\n",
			(unsigned long long)profile.jointDispatchCount, maxTransformError, maxVelocityError );
	ENSURE( profile.jointDispatchCount == 40 );
	ENSURE( profile.jointFallbackCount == 0 );
	ENSURE( maxTransformError <= 8.0e-4f );
	ENSURE( maxVelocityError <= 8.0e-4f );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalMixedJointOverflowTest( void )
{
	const int spokeCount = 32;
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -0.5f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.2f };
	b3BodyId cpuBodies[spokeCount + 1];
	b3BodyId gpuBodies[spokeCount + 1];
	b3BodyDef hubDef = b3DefaultBodyDef();
	hubDef.type = b3_dynamicBody;
	hubDef.enableSleep = false;
	hubDef.angularVelocity = (b3Vec3){ 0.04f, -0.03f, 0.02f };
	cpuBodies[0] = b3CreateBody( cpuWorld, &hubDef );
	gpuBodies[0] = b3CreateBody( gpuWorld, &hubDef );
	b3CreateSphereShape( cpuBodies[0], &shapeDef, &sphere );
	b3CreateSphereShape( gpuBodies[0], &shapeDef, &sphere );
	for ( int i = 0; i < spokeCount; ++i )
	{
		float angle = 2.0f * B3_PI * (float)i / (float)spokeCount;
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		bodyDef.position = (b3Pos){ 2.0f * cosf( angle ), 0.02f * (float)( i % 3 ), 2.0f * sinf( angle ) };
		bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 1.0f, 0.2f } ), 0.01f * (float)( i + 1 ) );
		bodyDef.angularVelocity = (b3Vec3){ -0.03f, 0.02f, 0.01f * (float)( i % 4 ) };
		cpuBodies[i + 1] = b3CreateBody( cpuWorld, &bodyDef );
		gpuBodies[i + 1] = b3CreateBody( gpuWorld, &bodyDef );
		b3CreateSphereShape( cpuBodies[i + 1], &shapeDef, &sphere );
		b3CreateSphereShape( gpuBodies[i + 1], &shapeDef, &sphere );
		if ( i & 1 )
		{
			b3ParallelJointDef cpuDef = b3DefaultParallelJointDef();
			cpuDef.base.bodyIdA = cpuBodies[0];
			cpuDef.base.bodyIdB = cpuBodies[i + 1];
			cpuDef.hertz = 2.5f;
			cpuDef.dampingRatio = 0.6f;
			cpuDef.maxTorque = 25.0f;
			b3CreateParallelJoint( cpuWorld, &cpuDef );
			b3ParallelJointDef gpuDef = cpuDef;
			gpuDef.base.bodyIdA = gpuBodies[0];
			gpuDef.base.bodyIdB = gpuBodies[i + 1];
			b3CreateParallelJoint( gpuWorld, &gpuDef );
		}
		else
		{
			b3DistanceJointDef cpuDef = b3DefaultDistanceJointDef();
			cpuDef.base.bodyIdA = cpuBodies[0];
			cpuDef.base.bodyIdB = cpuBodies[i + 1];
			cpuDef.length = 1.9f;
			b3CreateDistanceJoint( cpuWorld, &cpuDef );
			b3DistanceJointDef gpuDef = cpuDef;
			gpuDef.base.bodyIdA = gpuBodies[0];
			gpuDef.base.bodyIdB = gpuBodies[i + 1];
			b3CreateDistanceJoint( gpuWorld, &gpuDef );
		}
	}
	for ( int step = 0; step < 6; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxTransformError = 0.0f;
	float maxVelocityError = 0.0f;
	for ( int i = 0; i <= spokeCount; ++i )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.x - b.q.v.x ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.y - b.q.v.y ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.v.z - b.q.v.z ) );
		maxTransformError = b3MaxFloat( maxTransformError, fabsf( a.q.s - b.q.s ) );
		maxVelocityError = b3MaxFloat( maxVelocityError, (float)b3Length( b3Sub( b3Body_GetLinearVelocity( cpuBodies[i] ),
																				 b3Body_GetLinearVelocity( gpuBodies[i] ) ) ) );
		maxVelocityError = b3MaxFloat( maxVelocityError, (float)b3Length( b3Sub( b3Body_GetAngularVelocity( cpuBodies[i] ),
																				 b3Body_GetAngularVelocity( gpuBodies[i] ) ) ) );
	}
	b3Counters counters = b3World_GetCounters( cpuWorld );
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    mixed joint overflow=%d dispatches=%llu transformError=%.3g velocityError=%.3g\n",
			counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1], (unsigned long long)profile.jointDispatchCount, maxTransformError,
			maxVelocityError );
	ENSURE( counters.colorCounts[B3_GRAPH_COLOR_COUNT - 1] > 0 );
	ENSURE( profile.jointDispatchCount == 24 );
	ENSURE( profile.jointFallbackCount == 0 );
	ENSURE( maxTransformError <= 8.0e-4f );
	ENSURE( maxVelocityError <= 8.0e-4f );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

static int MetalStaticBodyJointTest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -3.0f, 0.0f };
	worldDef.enableSleep = false;
	b3WorldId cpuWorld = b3CreateWorld( &worldDef );
	b3WorldId gpuWorld = b3CreateWorld( &worldDef );
	ENSURE( b3World_EnableMetal( gpuWorld, 1 ) );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.filter.maskBits = 0;
	b3BoxHull hull = b3MakeBoxHull( 0.3f, 0.25f, 0.2f );
	b3BodyId cpuBodies[4];
	b3BodyId gpuBodies[4];
	for ( int i = 0; i < 2; ++i )
	{
		b3BodyDef staticDef = b3DefaultBodyDef();
		staticDef.position = (b3Pos){ 3.0f * (float)i, 0.0f, 0.0f };
		b3BodyDef dynamicDef = b3DefaultBodyDef();
		dynamicDef.type = b3_dynamicBody;
		dynamicDef.enableSleep = false;
		dynamicDef.position = (b3Pos){ 3.0f * (float)i + 1.2f, 0.2f, 0.0f };
		dynamicDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, 1.0f, 0.3f } ), 0.15f );
		dynamicDef.linearVelocity = (b3Vec3){ -0.1f, 0.05f, 0.02f };
		dynamicDef.angularVelocity = (b3Vec3){ 0.08f, -0.04f, 0.03f };
		cpuBodies[2 * i] = b3CreateBody( cpuWorld, &staticDef );
		cpuBodies[2 * i + 1] = b3CreateBody( cpuWorld, &dynamicDef );
		gpuBodies[2 * i] = b3CreateBody( gpuWorld, &staticDef );
		gpuBodies[2 * i + 1] = b3CreateBody( gpuWorld, &dynamicDef );
		b3CreateHullShape( cpuBodies[2 * i], &shapeDef, &hull.base );
		b3CreateHullShape( cpuBodies[2 * i + 1], &shapeDef, &hull.base );
		b3CreateHullShape( gpuBodies[2 * i], &shapeDef, &hull.base );
		b3CreateHullShape( gpuBodies[2 * i + 1], &shapeDef, &hull.base );
	}
	b3DistanceJointDef cpuDistance = b3DefaultDistanceJointDef();
	cpuDistance.base.bodyIdA = cpuBodies[0];
	cpuDistance.base.bodyIdB = cpuBodies[1];
	cpuDistance.length = 1.0f;
	b3CreateDistanceJoint( cpuWorld, &cpuDistance );
	b3DistanceJointDef gpuDistance = cpuDistance;
	gpuDistance.base.bodyIdA = gpuBodies[0];
	gpuDistance.base.bodyIdB = gpuBodies[1];
	b3CreateDistanceJoint( gpuWorld, &gpuDistance );
	b3ParallelJointDef cpuParallel = b3DefaultParallelJointDef();
	cpuParallel.base.bodyIdA = cpuBodies[2];
	cpuParallel.base.bodyIdB = cpuBodies[3];
	cpuParallel.hertz = 3.0f;
	cpuParallel.dampingRatio = 0.7f;
	cpuParallel.maxTorque = 20.0f;
	b3CreateParallelJoint( cpuWorld, &cpuParallel );
	b3ParallelJointDef gpuParallel = cpuParallel;
	gpuParallel.base.bodyIdA = gpuBodies[2];
	gpuParallel.base.bodyIdB = gpuBodies[3];
	b3CreateParallelJoint( gpuWorld, &gpuParallel );
	for ( int step = 0; step < 10; ++step )
	{
		b3World_Step( cpuWorld, 1.0f / 60.0f, 4 );
		b3World_Step( gpuWorld, 1.0f / 60.0f, 4 );
	}
	float maxError = 0.0f;
	for ( int i = 1; i < 4; i += 2 )
	{
		b3WorldTransform a = b3Body_GetTransform( cpuBodies[i] );
		b3WorldTransform b = b3Body_GetTransform( gpuBodies[i] );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.x - (double)b.p.x ) );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.y - (double)b.p.y ) );
		maxError = b3MaxFloat( maxError, (float)fabs( (double)a.p.z - (double)b.p.z ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.x - b.q.v.x ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.y - b.q.v.y ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.v.z - b.q.v.z ) );
		maxError = b3MaxFloat( maxError, fabsf( a.q.s - b.q.s ) );
	}
	b3MetalProfile profile = b3World_GetMetalProfile( gpuWorld );
	printf( "    static-body joints dispatches=%llu maxError=%.3g\n", (unsigned long long)profile.jointDispatchCount, maxError );
	ENSURE( profile.jointDispatchCount == 40 );
	ENSURE( profile.jointFallbackCount == 0 );
	ENSURE( maxError <= 8.0e-4f );
	b3DestroyWorld( gpuWorld );
	b3DestroyWorld( cpuWorld );
	return 0;
}

int MetalTest( void )
{
	RUN_SUBTEST( MetalPositionIntegrationTest );
	RUN_SUBTEST( MetalFusedIntegrationTest );
	RUN_SUBTEST( MetalFinalizationTest );
	RUN_SUBTEST( MetalConvexManifoldTest );
	RUN_SUBTEST( MetalResidentSolverOwnershipTest );
	RUN_SUBTEST( MetalContactMaterialCallbackExceptionTest );
	RUN_SUBTEST( MetalContactPreparePreSolveExceptionTest );
	RUN_SUBTEST( MetalContactPrepareFallbackTest );
	RUN_SUBTEST( MetalResidentContactPrepareDifferentialTest );
	RUN_SUBTEST( MetalResidentContactHitEventTest );
	RUN_SUBTEST( MetalResidentWarmStartCarryTest );
	RUN_SUBTEST( MetalPairTraversalTest );
	RUN_SUBTEST( MetalExistingPairFilterTest );
	RUN_SUBTEST( MetalPairTraversalFallbackTest );
	RUN_SUBTEST( MetalShapeCompactionTest );
	RUN_SUBTEST( MetalShapeInputRegistryTest );
	RUN_SUBTEST( MetalWorldIntegrationTest );
	RUN_SUBTEST( MetalDistanceJointContactTest );
	RUN_SUBTEST( MetalUnsupportedJointFallbackTest );
	RUN_SUBTEST( MetalConvexFrictionContactTest );
	RUN_SUBTEST( MetalConvexRestitutionContactTest );
	RUN_SUBTEST( MetalMeshContactTest );
	RUN_SUBTEST( MetalOverflowContactTest );
	RUN_SUBTEST( MetalDistanceJointModesAndOverflowTest );
	RUN_SUBTEST( MetalMixedDistanceParallelJointTest );
	RUN_SUBTEST( MetalMixedJointOverflowTest );
	RUN_SUBTEST( MetalStaticBodyJointTest );
	return 0;
}
