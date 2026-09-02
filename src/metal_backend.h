// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "body.h"
#include "solver.h"

#include <stdbool.h>

// Internal Apple GPU backend. The C boundary keeps Objective-C and Metal types
// out of the portable Box3D implementation.
typedef struct b3MetalContext b3MetalContext;
typedef struct b3ContactConstraintWide b3ContactConstraintWide;
typedef struct b3ContactConstraint b3ContactConstraint;
typedef struct b3ManifoldConstraint b3ManifoldConstraint;
typedef struct b3BroadPhase b3BroadPhase;
typedef struct b3World b3World;

typedef struct b3MetalPairQueryRecord
{
	uint32_t count;
	uint32_t offset;
	uint32_t flags;
	int queryShapeIndex;
	float lowerX, lowerY, lowerZ;
	float upperX, upperY, upperZ;
} b3MetalPairQueryRecord;

typedef struct b3MetalPairCandidate
{
	int proxyId;
	int treeType;
	int shapeIndex;
	int padding;
} b3MetalPairCandidate;

// Local sphere/capsule geometry in shape A's body frame. The record array is
// indexed exactly like the narrow-phase contact-index array. eligible == 0
// means the CPU path owns that contact; eligible != 0 is authoritative even
// when touching == 0.
typedef struct b3MetalConvexManifoldResult
{
	uint32_t eligible;
	uint32_t touching;
	uint32_t pointCount;
	uint32_t padding1;
	float normalX, normalY, normalZ;
	float padding2;
	float point1X, point1Y, point1Z, separation1;
	float point2X, point2Y, point2Z, separation2;
	uint32_t featureId1, featureId2;
	uint32_t padding3[2];
} b3MetalConvexManifoldResult;

typedef struct b3MetalDispatchStats
{
	double gpuMilliseconds;
	int bodyCount;
	int commandBufferCount;
	int treeUploadCount;
	int metadataUploadCount;
	int pairSetUploadCount;
} b3MetalDispatchStats;

// Returns false when there is no usable Metal device or the shader pipeline
// could not be compiled. errorBuffer may be NULL.
bool b3MetalCreateContext( b3MetalContext** context, char* errorBuffer, int errorCapacity );

void b3MetalDestroyContext( b3MetalContext* context );

// Copies the registry name of the selected GPU into nameBuffer.
void b3MetalGetDeviceName( const b3MetalContext* context, char* nameBuffer, int nameCapacity );

// Obtain persistent shared-storage memory for CPU contact preparation and Metal
// kernels. The pointer remains valid until this context grows the allocation or
// is destroyed.
b3ContactConstraintWide* b3MetalGetContactConstraintStorage( b3MetalContext* context, int constraintCount );
b3ContactConstraint* b3MetalGetMeshContactStorage( b3MetalContext* context, int constraintCount );
b3ManifoldConstraint* b3MetalGetMeshManifoldStorage( b3MetalContext* context, int manifoldCount );

// Integrates the complete awake-state array in one compute dispatch. The
// backend owns a persistent shared buffer so Apple unified memory is used and
// allocations are amortized across steps.
bool b3MetalIntegratePositions( b3MetalContext* context, b3BodyState* states, int bodyCount, float h,
								float maxLinearSpeed, float maxAngularSpeed, b3MetalDispatchStats* stats );

// Fused velocity and position integration for worlds with no active contacts
// or joints. Body properties are uploaded once and both stages execute in one
// dispatch, avoiding an intermediate CPU/GPU synchronization point.
bool b3MetalIntegrateUnconstrained( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims, int bodyCount,
									float h, b3Vec3 gravity, float maxLinearSpeed, float maxAngularSpeed,
									b3MetalDispatchStats* stats );

// Encode all unconstrained substeps into one command buffer. State and body
// properties are uploaded once, dispatches are separated by buffer barriers,
// and state is read back only after the final substep.
bool b3MetalIntegrateUnconstrainedSubsteps( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims,
	int bodyCount, int subStepCount, float h, b3Vec3 gravity, float maxLinearSpeed, float maxAngularSpeed,
	float invTimeStep, const b3MetalFinalizeResult** finalizeResults, b3StepContext* finalizationContext,
	b3MetalDispatchStats* stats );

// Execute velocity integration, colored convex/mesh contact and distance-joint
// solving, and position integration while state remains GPU-resident for every
// substep. Graph-overflow constraints execute serially in upstream order.
bool b3MetalSolveContactSubsteps( b3MetalContext* context, b3StepContext* stepContext,
	int velocityIterations, int relaxIterations, int restitutionIterations, b3MetalDispatchStats* stats );

// Compute transform, sleep-motion, and world-inertia finalization values. When
// statesAreResident is true, the persistent state buffer is reused without a
// CPU upload. results points into persistent shared storage owned by context.
bool b3MetalFinalizeBodies( b3MetalContext* context, const b3BodyState* states, const b3BodySim* sims,
	int bodyCount, float invTimeStep, bool statesAreResident, const b3MetalFinalizeResult** results,
	b3MetalDispatchStats* stats );

// Traverse the existing Box3D dynamic trees and compact candidate ranges on
// Metal. Per-move records and candidates preserve move-array and tree traversal
// order. Returns false before exposing results if the bounded Metal traversal
// cannot represent the step.
bool b3MetalGeneratePairCandidates( b3MetalContext* context, const b3World* world, const int* moveArray, int moveCount,
	const b3MetalPairQueryRecord** records,
	const b3MetalPairCandidate** candidates, int* candidateCount, b3MetalDispatchStats* stats );

// Batch the first common convex narrow-phase route. Results preserve contact
// array order and use exact VF64 subtraction for double-precision world
// translations before converting the relative displacement to float, matching
// Box3D's scalar narrow-phase boundary.
bool b3MetalComputeConvexManifolds( b3MetalContext* context, const b3World* world, const int* contactIndices,
	int contactCount, const b3MetalConvexManifoldResult** results, int* eligibleCount, b3MetalDispatchStats* stats );

// Mark the resident tree snapshot as matching CPU bounds after a successful
// shape-result leaf update/refit and the corresponding CPU bookkeeping pass.
void b3MetalCommitPairTreeRefit( b3MetalContext* context, const b3BroadPhase* broadPhase );

// Materialize current Metal-produced AABBs into the CPU shape mirror. The
// single-shape form is used by public queries; the all-shape form is a
// fail-closed boundary for route changes, fallback, and context destruction.
bool b3MetalSyncShapeBounds( b3MetalContext* context, b3World* world, int shapeId );
bool b3MetalSyncAllShapeBounds( b3MetalContext* context, b3World* world );
void b3MetalInvalidateShapeInputCache( b3MetalContext* context );
