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

typedef struct b3MetalDispatchStats
{
	double gpuMilliseconds;
	int bodyCount;
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
