// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "math_internal.h"
#include "simd.h"
#include "solver.h"

typedef struct b3Vec2W
{
	b3FloatW x, y;
} b3Vec2W;

typedef struct b3Vec3W
{
	b3FloatW X, Y, Z;
} b3Vec3W;

typedef struct b3QuatW
{
	b3Vec3W V;
	b3FloatW S;
} b3QuatW;

typedef struct b3SymMatrix2W
{
	b3FloatW cxx, cxy, cyy;
} b3SymMatrix2W;

typedef struct b3SymMatrix3W
{
	b3FloatW cxx, cxy, cxz, cyy, cyz, czz;
} b3SymMatrix3W;

typedef struct b3ContactConstraintPointWide
{
	b3Vec3W anchorAs, anchorBs;
	b3FloatW baseSeparations;
	b3FloatW normalImpulses;
	b3FloatW totalNormalImpulses;
	b3FloatW normalMasses;
	b3FloatW leverArms;
	b3FloatW relativeVelocities;
} b3ContactConstraintPointWide;

// Four convex contact constraints in structure-of-arrays form. This remains
// internal, but its layout is shared with the Metal solver buffer and guarded
// by ABI assertions in the backend.
typedef struct b3ContactConstraintWide
{
	int indexA[B3_SIMD_WIDTH];
	int indexB[B3_SIMD_WIDTH];
	int pointCounts[B3_SIMD_WIDTH];

	b3FloatW invMassA, invMassB;
	b3SymMatrix3W invIA, invIB;
	b3Vec3W normal;
	b3Vec3W tangent1;
	b3Vec3W tangent2;
	b3Vec3W centerA, centerB;
	b3FloatW twistMass;
	b3FloatW twistImpulse;
	b3SymMatrix2W tangentMass;
	b3Vec2W frictionImpulse;
	b3SymMatrix3W rollingMass;
	b3Vec3W rollingImpulse;
	b3FloatW friction;
	b3FloatW rollingResistance;
	b3FloatW tangentVelocity1;
	b3FloatW tangentVelocity2;
	b3FloatW biasRate;
	b3FloatW massScale;
	b3FloatW impulseScale;
	b3FloatW restitution;

	b3Manifold* manifolds[B3_SIMD_WIDTH];
	b3ContactConstraintPointWide points[B3_MAX_MANIFOLD_POINTS];
} b3ContactConstraintWide;

typedef struct b3ManifoldConstraintPoint
{
	b3Vec3 rA, rB;
	float baseSeparation;
	float relativeVelocity;
	float normalImpulse;
	float totalNormalImpulse;
	float normalMass;
	float leverArm;
} b3ManifoldConstraintPoint;

typedef struct b3ManifoldConstraint
{
	b3ManifoldConstraintPoint points[4];
	int pointCount;
	b3Vec3 normal;
	b3Vec3 tangent1;
	b3Vec3 tangent2;
	// Friction centers
	b3Vec3 centerA, centerB;
	float twistMass;
	float twistImpulse;
	b3Matrix2 tangentMass;
	b3Vec2 frictionImpulse;
	b3Vec3 rollingImpulse;
	float tangentVelocity1;
	float tangentVelocity2;
} b3ManifoldConstraint;

typedef struct b3ContactConstraint
{
	b3ManifoldConstraint* constraints;
	struct b3Contact* contact;
	int indexA;
	int indexB;
	float invMassA, invMassB;
	b3Matrix3 invIA, invIB;
	b3Softness softness;
	b3Matrix3 rollingMass;
	float friction;
	float restitution;
	float rollingResistance;
	int manifoldCount;
	int manifoldStart;
} b3ContactConstraint;

int b3GetWideContactConstraintByteCount( void );

// Overflow contacts don't fit into the constraint graph coloring
void b3PrepareContacts_Overflow( b3StepContext* context );
void b3WarmStartContacts_Overflow( b3StepContext* context );
void b3SolveContacts_Overflow( b3StepContext* context, bool useBias );
void b3ApplyRestitution_Overflow( b3StepContext* context );
void b3StoreImpulses_Overflow( b3StepContext* context );

void b3PrepareContacts_Mesh( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_Mesh( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_Mesh( b3SolverBlock block, b3StepContext* context, bool useBias );
void b3ApplyRestitution_Mesh( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_Mesh( b3SolverBlock block, b3StepContext* context, int workerIndex );

void b3PrepareContacts_Convex( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_Convex( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_Convex( b3SolverBlock block, b3StepContext* context, bool useBias );
void b3ApplyRestitution_Convex( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_Convex( b3SolverBlock block, b3StepContext* context, int workerIndex );
