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
typedef struct b3Contact b3Contact;

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

// Finalized sphere/capsule/compact-hull geometry in world axes, with point
// anchors relative to each body's center of mass. The record also carries GPU-authored point
// persistence and warm-start impulses matched against the prior resident solve.
// Compact records preserve input order.
// eligible == 0 means the CPU path owns that contact; eligible != 0 is authoritative even
// when touching == 0.
typedef struct b3MetalConvexManifoldResult
{
	uint32_t eligible;
	uint32_t touching;
	uint32_t pointCount;
	uint32_t inputIndex;
	float normalX, normalY, normalZ;
	uint32_t contactGeneration;
	float point1X, point1Y, point1Z, separation1;
	float point2X, point2Y, point2Z, separation2;
	uint32_t featureId1, featureId2;
	uint32_t scanOffset;
	uint32_t contactId;
	float normalImpulse1, normalImpulse2;
	uint32_t persistedBits;
	uint32_t residentFlags;
	float friction, restitution, rollingResistance, materialPadding;
	float tangentVelocityX, tangentVelocityY, tangentVelocityZ, tangentVelocityPadding;
	float anchorB1X, anchorB1Y, anchorB1Z, anchorB1Padding;
	float anchorB2X, anchorB2Y, anchorB2Z, anchorB2Padding;
} b3MetalConvexManifoldResult;

// Compact post-solve state written by Metal and indexed by contact id. This is
// the CPU/public-manifold synchronization boundary; it avoids rereading the
// complete SIMD-wide solver record after the command buffer completes.
typedef struct b3MetalContactImpulseResult
{
	uint32_t contactId;
	uint32_t generation;
	uint32_t pointCount;
	uint32_t contactGeneration;
	float frictionX, frictionY, frictionZ, twistImpulse;
	float rollingX, rollingY, rollingZ, padding;
	struct
	{
		float normalImpulse, totalNormalImpulse, normalVelocity;
		uint32_t featureId;
	} points[2];
} b3MetalContactImpulseResult;

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
bool b3MetalIntegratePositions( b3MetalContext* context, b3BodyState* states, int bodyCount, float h, float maxLinearSpeed,
								float maxAngularSpeed, b3MetalDispatchStats* stats );

// Fused velocity and position integration for worlds with no active contacts
// or joints. Body properties are uploaded once and both stages execute in one
// dispatch, avoiding an intermediate CPU/GPU synchronization point.
bool b3MetalIntegrateUnconstrained( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims, int bodyCount, float h,
									b3Vec3 gravity, float maxLinearSpeed, float maxAngularSpeed, b3MetalDispatchStats* stats );

// Encode all unconstrained substeps into one command buffer. State and body
// properties are uploaded once, dispatches are separated by buffer barriers,
// and state is read back only after the final substep.
bool b3MetalIntegrateUnconstrainedSubsteps( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims, int bodyCount,
											int subStepCount, float h, b3Vec3 gravity, float maxLinearSpeed,
											float maxAngularSpeed, float invTimeStep,
											const b3MetalFinalizeResult** finalizeResults, b3StepContext* finalizationContext,
											b3MetalDispatchStats* stats );

// Execute velocity integration, colored convex/mesh contact and distance-joint
// solving, and position integration while state remains GPU-resident for every
// substep. Graph-overflow constraints execute serially in upstream order.
bool b3MetalSolveContactSubsteps( b3MetalContext* context, b3StepContext* stepContext, int velocityIterations,
								  int relaxIterations, int restitutionIterations, b3MetalDispatchStats* stats );

// Compute transform, sleep-motion, and world-inertia finalization values. When
// statesAreResident is true, the persistent state buffer is reused without a
// CPU upload. results points into persistent shared storage owned by context.
bool b3MetalFinalizeBodies( b3MetalContext* context, const b3BodyState* states, const b3BodySim* sims, int bodyCount,
							float invTimeStep, bool statesAreResident, const b3MetalFinalizeResult** results,
							b3MetalDispatchStats* stats );

// Traverse the existing Box3D dynamic trees and compact candidate ranges on
// Metal. Per-move records and candidates preserve move-array and tree traversal
// order. Returns false before exposing results if the bounded Metal traversal
// cannot represent the step.
bool b3MetalGeneratePairCandidates( b3MetalContext* context, const b3World* world, const int* moveArray, int moveCount,
									const b3MetalPairQueryRecord** records, const b3MetalPairCandidate** candidates,
									int* candidateCount, b3MetalDispatchStats* stats );

// Batch the first common convex narrow-phase route. The returned array contains
// only active Metal results, ordered by inputIndex; eligibleCount is its length.
// Result normals and anchors are oriented into world axes and relative to each
// body's center of mass. Exact VF64 subtraction is used for double-precision
// world translations before converting the relative displacement to float,
// matching Box3D's scalar narrow-phase boundary.
bool b3MetalComputeConvexManifolds( b3MetalContext* context, const b3World* world, const int* contactIndices, int contactCount,
									const b3MetalConvexManifoldResult** results, int* eligibleCount,
									b3MetalDispatchStats* stats );

// Explicit diagnostic/fallback staging of the private contact-id-indexed table.
// This submits and waits for a blit; it is not part of the steady narrow phase.
bool b3MetalCopyResidentConvexManifoldTable( b3MetalContext* context, b3MetalConvexManifoldResult* results, int resultCapacity );

// Materialize current private manifold geometry into the CPU mirror. The
// single-contact form is the lazy public boundary; the all-contact form is used
// before route changes and CPU solver fallback.
bool b3MetalSyncContactManifold( b3MetalContext* context, b3Contact* contact );
bool b3MetalSyncAllContactManifolds( b3MetalContext* context, b3World* world );

// Retain post-persistence solver metadata by contact id during the existing CPU
// collision pass. The narrow-phase dispatch preallocates the table, so parallel
// collision workers only write disjoint records and never allocate.
bool b3MetalStageResidentContactPrepare( b3MetalContext* context, b3Contact* contact );

// Return the current shared compact post-solve table. A result is authoritative
// only when its contactId and generation match the requested entry.
const b3MetalContactImpulseResult* b3MetalGetResidentContactImpulseTable( const b3MetalContext* context, uint32_t* generation,
																		  int* resultCount );

// Return the contact ids whose shapes requested hit events during the current
// resident narrow-phase input pass. The list is already compact and is valid
// only while the latest post-solve result table is authoritative.
const int* b3MetalGetResidentHitEventContacts( const b3MetalContext* context, int* contactCount );

// Materialize one contact's GPU-authored impulse state into its CPU/public
// manifold. Contact generation and point feature ids are validated before any
// field is changed.
bool b3MetalSyncContactImpulses( const b3MetalContext* context, b3Contact* contact );

// Drop authority from a prior GPU solve after its warm-start state has been
// consumed by the next collision pass and before a new solver route is chosen.
void b3MetalInvalidateContactImpulseResults( b3MetalContext* context );

// Mark the resident tree snapshot as matching CPU bounds after a successful
// shape-result leaf update/refit and the corresponding CPU bookkeeping pass.
void b3MetalCommitPairTreeRefit( b3MetalContext* context, const b3BroadPhase* broadPhase );

// Materialize current Metal-produced AABBs into the CPU shape mirror. The
// single-shape form is used by public queries; the all-shape form is a
// fail-closed boundary for route changes, fallback, and context destruction.
bool b3MetalSyncShapeBounds( b3MetalContext* context, b3World* world, int shapeId );
bool b3MetalSyncAllShapeBounds( b3MetalContext* context, b3World* world );
void b3MetalInvalidateShapeInputCache( b3MetalContext* context );
