// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"

#include "constraint_graph.h"
#include "broad_phase.h"
#include "contact_solver.h"
#include "hull.h"
#include "joint.h"
#include "physics_world.h"
#include "shape.h"
#include "solver_set.h"

#include "box3d/constants.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined( BOX3D_DOUBLE_PRECISION )
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#include "metal_vf64_ieee_source.h"
#pragma clang diagnostic pop
#endif

// These checks protect the shared C/MSL ABI. Metal's native float3 is 16-byte
// aligned, so the shader deliberately uses scalar fields matching this layout.
_Static_assert( sizeof( b3BodyState ) == 56, "Metal body-state ABI changed" );
_Static_assert( offsetof( b3BodyState, angularVelocity ) == 12, "Metal body-state ABI changed" );
_Static_assert( offsetof( b3BodyState, deltaPosition ) == 24, "Metal body-state ABI changed" );
_Static_assert( offsetof( b3BodyState, deltaRotation ) == 36, "Metal body-state ABI changed" );
_Static_assert( offsetof( b3BodyState, flags ) == 52, "Metal body-state ABI changed" );
_Static_assert( B3_GYROSCOPIC_ITERATIONS == 1, "Metal fused integration must match the configured gyro iteration count" );

struct b3MetalContext
{
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	id<MTLComputePipelineState> integratePositionsPipeline;
	id<MTLComputePipelineState> integrateUnconstrainedPipeline;
	id<MTLComputePipelineState> finalizeBodiesPipeline;
	id<MTLComputePipelineState> finalizeShapesPipeline;
	id<MTLComputePipelineState> shapeScanBlocksPipeline;
	id<MTLComputePipelineState> shapePrefixPipeline;
	id<MTLComputePipelineState> shapeScatterPipeline;
	id<MTLComputePipelineState> pairCandidatesPipeline;
	id<MTLComputePipelineState> pairScanBlocksPipeline;
	id<MTLComputePipelineState> pairPrefixPipeline;
	id<MTLComputePipelineState> pairAddOffsetsPipeline;
	id<MTLComputePipelineState> pairUpdateLeavesPipeline;
	id<MTLComputePipelineState> pairRefitPipeline;
	id<MTLComputePipelineState> convexManifoldPipeline;
	id<MTLComputePipelineState> convexManifoldScanPipeline;
	id<MTLComputePipelineState> convexManifoldPrefixPipeline;
	id<MTLComputePipelineState> convexManifoldScatterPipeline;
	id<MTLComputePipelineState> prepareContactsPipeline;
	id<MTLComputePipelineState> storeContactImpulsesPipeline;
	id<MTLComputePipelineState> warmStartContactsPipeline;
	id<MTLComputePipelineState> solveContactsPipeline;
	id<MTLComputePipelineState> restitutionContactsPipeline;
	id<MTLComputePipelineState> warmStartMeshPipeline;
	id<MTLComputePipelineState> solveMeshPipeline;
	id<MTLComputePipelineState> restitutionMeshPipeline;
	id<MTLComputePipelineState> warmStartOverflowPipeline;
	id<MTLComputePipelineState> solveOverflowPipeline;
	id<MTLComputePipelineState> restitutionOverflowPipeline;
	id<MTLComputePipelineState> warmStartDistancePipeline;
	id<MTLComputePipelineState> solveDistancePipeline;
	id<MTLComputePipelineState> warmStartParallelPipeline;
	id<MTLComputePipelineState> solveParallelPipeline;
	id<MTLComputePipelineState> warmStartJointOverflowPipeline;
	id<MTLComputePipelineState> solveJointOverflowPipeline;
	id<MTLBuffer> bodyStateBuffer;
	NSUInteger bodyStateCapacity;
	int bodyStateResidentCount;
	uint64_t bodyStateResidentRevision;
	id<MTLBuffer> bodyPropertiesBuffer;
	NSUInteger bodyPropertiesCapacity;
	int bodyPropertiesResidentCount;
	uint64_t bodyPropertiesResidentRevision;
	id<MTLBuffer> finalizeResultBuffer;
	NSUInteger finalizeResultCapacity;
	id<MTLBuffer> finalizeReadbackBuffer;
	NSUInteger finalizeReadbackCapacity;
	id<MTLBuffer> finalizePropertiesBuffer;
	NSUInteger finalizePropertiesCapacity;
	id<MTLBuffer> bodyMoveResultBuffer;
	NSUInteger bodyMoveResultCapacity;
	id<MTLBuffer> bodyMoveReadbackBuffer;
	NSUInteger bodyMoveReadbackCapacity;
	int bodyMoveResultCount;
	uint64_t bodyMoveResultStepIndex;
	id<MTLBuffer> shapeInputBuffer;
	NSUInteger shapeInputCapacity;
	int* shapeInputBodyIds;
	int shapeInputBodyCapacity;
	int shapeInputBodyCount;
	int shapeInputCount;
	bool shapeInputAllMasksDisabled;
	bool shapeInputCacheValid;
	id<MTLBuffer> shapeResultBuffer;
	NSUInteger shapeResultCapacity;
	id<MTLBuffer> shapeReadbackBuffer;
	NSUInteger shapeReadbackCapacity;
	uint64_t shapeBoundsRevision;
	int shapeBoundsCount;
	id<MTLBuffer> shapeCompactBuffer;
	NSUInteger shapeCompactCapacity;
	id<MTLBuffer> shapeBlockBuffer;
	NSUInteger shapeBlockCapacity;
	id<MTLBuffer> shapeSummaryBuffer;
	id<MTLBuffer> pairMoveBuffer;
	NSUInteger pairMoveCapacity;
	id<MTLBuffer> pairTreeBuffer;
	NSUInteger pairTreeCapacity;
	id<MTLBuffer> pairMovedBuffer;
	NSUInteger pairMovedCapacity;
	id<MTLBuffer> pairShapeBuffer;
	NSUInteger pairShapeCapacity;
	uint64_t pairShapeRevision;
	id<MTLBuffer> pairSetBuffer;
	NSUInteger pairSetCapacity;
	uint64_t pairSetRevision;
	uint64_t pairTreeRevision;
	uint32_t pairTreeOffsets[b3_bodyTypeCount];
	uint32_t pairTreeNodeCounts[b3_bodyTypeCount];
	uint16_t pairTreeHeights[b3_bodyTypeCount];
	id<MTLBuffer> pairRecordBuffer;
	NSUInteger pairRecordCapacity;
	id<MTLBuffer> pairCandidateBuffer;
	NSUInteger pairCandidateCapacity;
	id<MTLBuffer> pairSummaryBuffer;
	id<MTLBuffer> pairBlockBuffer;
	NSUInteger pairBlockCapacity;
	id<MTLBuffer> convexManifoldInputBuffer;
	NSUInteger convexManifoldInputCapacity;
	int convexManifoldInputCount;
	int convexManifoldCandidateCount;
	uint64_t convexManifoldInputPairRevision;
	uint64_t convexManifoldInputGraphRevision;
	uint64_t convexManifoldInputRevision;
	id<MTLBuffer> convexManifoldResultBuffer;
	NSUInteger convexManifoldResultCapacity;
	id<MTLBuffer> convexManifoldCompactBuffer;
	NSUInteger convexManifoldCompactCapacity;
	id<MTLBuffer> convexManifoldBlockBuffer;
	NSUInteger convexManifoldBlockCapacity;
	id<MTLBuffer> convexManifoldSummaryBuffer;
	id<MTLBuffer> convexManifoldTableBuffer;
	NSUInteger convexManifoldTableCapacity;
	int convexManifoldTableCount;
	id<MTLBuffer> convexManifoldTableReadbackBuffer;
	NSUInteger convexManifoldTableReadbackCapacity;
	id<MTLBuffer> convexHullPointBuffer;
	NSUInteger convexHullPointCapacity;
	id<MTLBuffer> convexHullPlaneBuffer;
	NSUInteger convexHullPlaneCapacity;
	id<MTLBuffer> convexHullTriangleBuffer;
	NSUInteger convexHullTriangleCapacity;
	id<MTLBuffer> convexShapeGeometryBuffer;
	NSUInteger convexShapeGeometryCapacity;
	int convexShapeGeometryCount;
	uint64_t convexShapeGeometryRevision;
	uint64_t convexShapeMaterialRevision;
	id<MTLBuffer> convexBodyTransformBuffer;
	NSUInteger convexBodyTransformCapacity;
	int convexBodyTransformCount;
	uint64_t convexBodyTransformStepIndex;
	uint64_t convexBodyTransformRevision;
	id<MTLBuffer> contactConstraintBuffer;
	NSUInteger contactConstraintCapacity;
	id<MTLBuffer> contactPrepareTableBuffer;
	NSUInteger contactPrepareTableCapacity;
	id<MTLBuffer> contactPrepareIndexBuffer;
	NSUInteger contactPrepareIndexCapacity;
	uint64_t contactPrepareScheduleRevision;
	int contactPrepareScheduleWideCount;
	int contactPrepareScheduleContactCount;
	id<MTLBuffer> contactPrepareStatusBuffer;
	uint32_t contactPrepareGeneration;
	id<MTLBuffer> contactImpulseResultBuffer;
	NSUInteger contactImpulseResultCapacity;
	int contactImpulseResultCount;
	uint32_t contactImpulseResultGeneration;
	id<MTLBuffer> contactHitEventIdBuffer;
	NSUInteger contactHitEventIdCapacity;
	int contactHitEventIdCount;
	id<MTLBuffer> meshContactBuffer;
	NSUInteger meshContactCapacity;
	id<MTLBuffer> meshManifoldBuffer;
	NSUInteger meshManifoldCapacity;
	id<MTLBuffer> distanceJointBuffer;
	NSUInteger distanceJointCapacity;
	id<MTLBuffer> parallelJointBuffer;
	NSUInteger parallelJointCapacity;
	id<MTLBuffer> jointOverflowBuffer;
	NSUInteger jointOverflowCapacity;
};

typedef struct b3MetalIntegrateParams
{
	uint32_t bodyCount;
	float h;
	float maxLinearSpeed;
	float maxAngularSpeed;
} b3MetalIntegrateParams;

// Compact, stable transfer format. b3BodySim contains collision/finalization
// fields the integration kernel never reads; excluding them saves bandwidth
// and also keeps this ABI independent of BOX3D_DOUBLE_PRECISION.
typedef struct b3MetalBodyProperties
{
	float qx, qy, qz, qw;
	float forceX, forceY, forceZ;
	float torqueX, torqueY, torqueZ;
	float invMass;
	float invInertiaLocal[9];
	float invInertiaWorld[9];
	float linearDamping, angularDamping, gravityScale;
} b3MetalBodyProperties;

typedef struct b3MetalFinalizeProperties
{
	float localCenterX, localCenterY, localCenterZ;
	float maxExtentX, maxExtentY, maxExtentZ;
	float centerX, centerY, centerZ;
	int32_t bodyId;
	uint64_t centerXBits, centerYBits, centerZBits;
	uint64_t userData;
	uint32_t generationWorld;
	uint32_t padding;
} b3MetalFinalizeProperties;

typedef struct b3MetalBodyMoveResult
{
	float qx, qy, qz, qw;
	float px, py, pz;
	int32_t bodyId;
	uint64_t pxBits, pyBits, pzBits;
	uint64_t userData;
	uint32_t generationWorld;
	uint32_t padding;
} b3MetalBodyMoveResult;

typedef struct b3MetalShapeInput
{
	uint32_t bodyIndex;
	uint32_t shapeId;
	uint32_t type;
	int32_t proxyKey;
	float point1X, point1Y, point1Z, radius;
	float point2X, point2Y, point2Z, margin;
	float fatLowerX, fatLowerY, fatLowerZ;
	float fatUpperX, fatUpperY, fatUpperZ;
} b3MetalShapeInput;

typedef struct b3MetalPairSummary
{
	uint64_t totalCount;
	uint32_t flags;
	uint32_t writeFlags;
} b3MetalPairSummary;

typedef struct b3MetalPairBlock
{
	uint32_t sum;
	uint32_t flags;
	uint32_t offset;
	uint32_t padding;
} b3MetalPairBlock;

typedef struct b3MetalPairShape
{
	int32_t bodyId;
	int32_t sensorIndex;
	int32_t groupIndex;
	uint32_t type;
	uint64_t categoryBits;
	uint64_t maskBits;
} b3MetalPairShape;

typedef struct b3MetalConvexManifoldInput
{
	uint32_t eligible;
	uint32_t shapeIdA, shapeIdB, contactId;
	uint32_t contactGeneration;
	uint32_t prepareEligible;
	int32_t indexA, indexB;
} b3MetalConvexManifoldInput;

typedef struct b3MetalBodyTransform
{
	float qx, qy, qz, qw;
	float px, py, pz;
	uint32_t supported;
	uint64_t pxBits, pyBits, pzBits;
	int32_t index;
	uint32_t flags;
	float localCenterX, localCenterY, localCenterZ, centerPadding;
} b3MetalBodyTransform;

typedef struct b3MetalHullTriangle
{
	uint32_t index1, index2, index3, face;
} b3MetalHullTriangle;

typedef struct b3MetalFloat4
{
	float x, y, z, w;
} b3MetalFloat4;

typedef struct b3MetalShapeGeometry
{
	float point1X, point1Y, point1Z, radius;
	float point2X, point2Y, point2Z;
	int32_t bodyId;
	uint32_t pointOffset, pointCount;
	uint32_t planeOffset, planeCount;
	uint32_t triangleOffset, triangleCount;
	uint32_t type, supported;
	float friction, restitution, rollingResistance, rollingRadius;
	float tangentVelocityX, tangentVelocityY, tangentVelocityZ, materialPadding;
} b3MetalShapeGeometry;

_Static_assert( sizeof( b3MetalBodyProperties ) == 128, "Metal body property ABI changed" );
_Static_assert( sizeof( b3MetalFinalizeProperties ) == 80, "Metal finalization-property ABI changed" );
_Static_assert( sizeof( b3MetalFinalizeResult ) == 100, "Metal finalization-result ABI changed" );
_Static_assert( sizeof( b3MetalBodyMoveResult ) == 72, "Metal body-move result ABI changed" );
_Static_assert( sizeof( b3MetalShapeInput ) == 72, "Metal shape-input ABI changed" );
_Static_assert( sizeof( b3MetalShapeAABBResult ) == 64, "Metal shape-result ABI changed" );
_Static_assert( sizeof( b3MetalEnlargedShapeResult ) == 32, "Metal enlarged-shape ABI changed" );
_Static_assert( sizeof( b3TreeNode ) == 48, "Metal tree-node ABI changed" );
_Static_assert( offsetof( b3TreeNode, categoryBits ) == 24, "Metal tree-node ABI changed" );
_Static_assert( offsetof( b3TreeNode, children ) == 32, "Metal tree-node ABI changed" );
_Static_assert( offsetof( b3TreeNode, parent ) == 40, "Metal tree-node ABI changed" );
_Static_assert( offsetof( b3TreeNode, flags ) == 46, "Metal tree-node ABI changed" );
_Static_assert( sizeof( b3MetalPairQueryRecord ) == 40, "Metal pair-record ABI changed" );
_Static_assert( sizeof( b3MetalPairCandidate ) == 16, "Metal pair-candidate ABI changed" );
_Static_assert( sizeof( b3MetalPairSummary ) == 16, "Metal pair-summary ABI changed" );
_Static_assert( sizeof( b3MetalPairBlock ) == 16, "Metal pair-block ABI changed" );
_Static_assert( sizeof( b3MetalPairShape ) == 32, "Metal pair-shape ABI changed" );
_Static_assert( sizeof( b3MetalConvexManifoldInput ) == 32, "Metal convex-manifold input ABI changed" );
_Static_assert( sizeof( b3MetalBodyTransform ) == 80, "Metal body-transform ABI changed" );
_Static_assert( sizeof( b3MetalConvexManifoldResult ) == 160, "Metal convex-manifold result ABI changed" );
_Static_assert( offsetof( b3MetalConvexManifoldResult, inputIndex ) == 12, "Metal manifold input-index ABI changed" );
_Static_assert( offsetof( b3MetalConvexManifoldResult, scanOffset ) == 72, "Metal manifold scan-offset ABI changed" );
_Static_assert( offsetof( b3MetalConvexManifoldResult, contactId ) == 76, "Metal manifold contact-id ABI changed" );
_Static_assert( sizeof( b3MetalHullTriangle ) == 16, "Metal hull-triangle ABI changed" );
_Static_assert( sizeof( b3MetalFloat4 ) == 16, "Metal float4 ABI changed" );
_Static_assert( sizeof( b3MetalShapeGeometry ) == 96, "Metal shape-geometry record ABI changed" );
_Static_assert( sizeof( b3SetItem ) == 16, "Metal pair-set item ABI changed" );
_Static_assert( sizeof( b3ContactConstraintPointWide ) == 192, "Metal wide contact point ABI changed" );
_Static_assert( sizeof( b3ContactConstraintWide ) == 1696, "Metal wide contact ABI changed" );
_Static_assert( sizeof( b3ManifoldConstraintPoint ) == 48, "Metal mesh contact-point ABI changed" );
_Static_assert( sizeof( b3ManifoldConstraint ) == 308, "Metal mesh manifold ABI changed" );
_Static_assert( sizeof( b3ContactConstraint ) == 176, "Metal mesh contact ABI changed" );

typedef struct b3MetalFusedParams
{
	uint32_t bodyCount;
	float h;
	float maxLinearSpeed;
	float maxAngularSpeed;
	float gravityX, gravityY, gravityZ;
	uint32_t integratePosition;
} b3MetalFusedParams;

typedef struct b3MetalContactParams
{
	uint32_t offset;
	uint32_t count;
	float invH;
	float contactSpeed;
	uint32_t useBias;
	float restitutionThreshold;
	uint32_t padding[2];
} b3MetalContactParams;

_Static_assert( sizeof( b3MetalContactParams ) == 32, "Metal contact parameter ABI changed" );

typedef struct b3MetalContactPreparePoint
{
	float anchorAX, anchorAY, anchorAZ, separation;
	float anchorBX, anchorBY, anchorBZ, normalImpulse;
	uint32_t featureId;
} b3MetalContactPreparePoint;

typedef struct b3MetalContactPrepareInput
{
	uint32_t contactId;
	int indexA, indexB;
	uint32_t generation;
	uint64_t manifold;
	float friction, restitution;
	float rollingResistance, tangentVelocityX, tangentVelocityY, tangentVelocityZ;
	float twistImpulse, frictionImpulseX, frictionImpulseY, frictionImpulseZ;
	float rollingImpulseX, rollingImpulseY, rollingImpulseZ;
	uint32_t contactGeneration;
	b3MetalContactPreparePoint points[2];
} b3MetalContactPrepareInput;

typedef struct b3MetalContactPrepareParams
{
	uint32_t wideCount, tableCount;
	float warmStartScale, invTau;
	b3Softness contactSoftness;
	float padding0;
	b3Softness staticSoftness;
	uint32_t generation;
} b3MetalContactPrepareParams;

_Static_assert( sizeof( b3MetalContactPreparePoint ) == 36, "Metal contact-prepare point ABI changed" );
_Static_assert( sizeof( b3MetalContactPrepareInput ) == 152, "Metal contact-prepare input ABI changed" );
_Static_assert( sizeof( b3MetalContactPrepareParams ) == 48, "Metal contact-prepare parameter ABI changed" );
_Static_assert( sizeof( b3MetalContactImpulseResult ) == 80, "Metal contact-impulse result ABI changed" );

typedef struct b3MetalContactImpulseParams
{
	uint32_t wideCount, tableCount, generation, padding;
} b3MetalContactImpulseParams;

_Static_assert( sizeof( b3MetalContactImpulseParams ) == 16, "Metal contact-impulse parameter ABI changed" );

typedef struct b3MetalDistanceJoint
{
	int indexA, indexB;
	float invMassA, invMassB;
	b3Matrix3 invIA, invIB;
	b3Softness constraintSoftness;
	b3Vec3 anchorA, anchorB, deltaCenter;
	b3Softness distanceSoftness;
	float length, hertz, lowerSpringForce, upperSpringForce;
	float minLength, maxLength, maxMotorForce, motorSpeed;
	float impulse, lowerImpulse, upperImpulse, motorImpulse, axialMass;
	uint32_t flags;
} b3MetalDistanceJoint;

typedef struct b3MetalJointParams
{
	uint32_t offset, count;
	float h, invH;
	uint32_t useBias;
	uint32_t padding[3];
} b3MetalJointParams;

typedef struct b3MetalParallelJoint
{
	int indexA, indexB;
	b3Matrix3 invIA, invIB;
	b3Softness softness;
	b3Vec3 perpAxisX, perpAxisY;
	b3Quat quatA, quatB;
	float maxTorque;
	b3Vec2 perpImpulse;
	uint32_t fixedRotation;
} b3MetalParallelJoint;

typedef struct b3MetalJointOverflow
{
	uint32_t type;
	uint32_t index;
} b3MetalJointOverflow;

_Static_assert( sizeof( b3MetalDistanceJoint ) == 204, "Metal distance-joint ABI changed" );
_Static_assert( offsetof( b3MetalDistanceJoint, invIA ) == 16, "Metal distance-joint ABI changed" );
_Static_assert( offsetof( b3MetalDistanceJoint, constraintSoftness ) == 88, "Metal distance-joint ABI changed" );
_Static_assert( offsetof( b3MetalDistanceJoint, anchorA ) == 100, "Metal distance-joint ABI changed" );
_Static_assert( offsetof( b3MetalDistanceJoint, distanceSoftness ) == 136, "Metal distance-joint ABI changed" );
_Static_assert( offsetof( b3MetalDistanceJoint, flags ) == 200, "Metal distance-joint ABI changed" );
_Static_assert( sizeof( b3MetalJointParams ) == 32, "Metal joint parameter ABI changed" );
_Static_assert( sizeof( b3MetalParallelJoint ) == 164, "Metal parallel-joint ABI changed" );
_Static_assert( offsetof( b3MetalParallelJoint, softness ) == 80, "Metal parallel-joint ABI changed" );
_Static_assert( offsetof( b3MetalParallelJoint, quatA ) == 116, "Metal parallel-joint ABI changed" );
_Static_assert( offsetof( b3MetalParallelJoint, fixedRotation ) == 160, "Metal parallel-joint ABI changed" );
_Static_assert( sizeof( b3MetalJointOverflow ) == 8, "Metal joint-overflow ABI changed" );

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
static const char* b3_metalSource =
	"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"struct BodyState {\n"
	"  float lvx, lvy, lvz;\n"
	"  float avx, avy, avz;\n"
	"  float dpx, dpy, dpz;\n"
	"  float qx, qy, qz, qw;\n"
	"  uint flags;\n"
	"};\n"
	"struct Params { uint bodyCount; float h; float maxLinearSpeed; float maxAngularSpeed; };\n"
	"struct BodyProperties {\n"
	"  float qx, qy, qz, qw;\n"
	"  float forceX, forceY, forceZ;\n"
	"  float torqueX, torqueY, torqueZ;\n"
	"  float invMass;\n"
	"  float invInertiaLocal[9];\n"
	"  float invInertiaWorld[9];\n"
	"  float linearDamping, angularDamping, gravityScale;\n"
	"};\n"
	"struct FinalizeProperties {\n"
	"  float localCenterX, localCenterY, localCenterZ;\n"
	"  float maxExtentX, maxExtentY, maxExtentZ;\n"
	"  float centerX, centerY, centerZ; int bodyId;\n"
	"  ulong centerXBits, centerYBits, centerZBits;\n"
	"  ulong userData; uint generationWorld,padding;\n"
	"};\n"
	"struct FinalizeResult {\n"
	"  float dpx, dpy, dpz; float qx, qy, qz, qw;\n"
	"  float originX, originY, originZ;\n"
	"  float positionX, positionY, positionZ;\n"
	"  float sleepVelocity, maxVelocity, maxDeltaPosition;\n"
	"  float invInertiaWorld[9];\n"
	"};\n"
	"struct BodyMoveResult {\n"
	"  float qx,qy,qz,qw,px,py,pz; int bodyId;\n"
	"  ulong pxBits,pyBits,pzBits,userData; uint generationWorld,padding;\n"
	"};\n"
	"struct ShapeInput {\n"
	"  uint bodyIndex, shapeId, type; int proxyKey;\n"
	"  float p1x,p1y,p1z,radius; float p2x,p2y,p2z,margin;\n"
	"  float oldLx,oldLy,oldLz,oldUx,oldUy,oldUz;\n"
	"};\n"
	"struct ShapeResult {\n"
	"  uint shapeId, bodyIndex, enlarged, padding;\n"
	"  float lx,ly,lz,ux,uy,uz; float flx,fly,flz,fux,fuy,fuz;\n"
	"};\n"
	"struct EnlargedShapeResult { int shapeId,proxyKey; float lx,ly,lz,ux,uy,uz; };\n"
	"struct TreeNode {\n"
	"  float lx,ly,lz,ux,uy,uz; ulong categoryBits; uint child1,child2; int parent; ushort height,flags;\n"
	"};\n"
	"struct PairQueryRecord { uint count,offset,flags; int queryShapeIndex; float lx,ly,lz,ux,uy,uz; };\n"
	"struct PairCandidate { int proxyId,treeType,shapeIndex,padding; };\n"
	"struct PairShape { int bodyId,sensorIndex,groupIndex; uint type; ulong categoryBits,maskBits; };\n"
	"struct SetItem { ulong key; uint hash,padding; };\n"
	"struct PairSummary { ulong totalCount; uint flags,writeFlags; };\n"
	"struct PairBlock { uint sum,flags,offset,padding; };\n"
	"struct PairParams { int root0,root1,root2; uint offset0,offset1,offset2,moveCount,writeCandidates,shapeCount,pairCapacity,p1,p2; };\n"
	"struct PairPrefixParams { uint moveCount,candidateCapacity,candidateLimit,padding; };\n"
	"struct ConvexManifoldInput {\n"
	"  uint eligible,shapeIdA,shapeIdB,contactId,contactGeneration,prepareEligible; int indexA,indexB;\n"
	"};\n"
	"struct BodyTransform { float qx,qy,qz,qw,px,py,pz; uint supported; ulong pxBits,pyBits,pzBits; int index; uint flags;\n"
	"  float localCenterX,localCenterY,localCenterZ,centerPadding; };\n"
	"struct HullTriangle { uint index1,index2,index3,face; };\n"
	"struct ShapeGeometry { float point1X,point1Y,point1Z,radius; float point2X,point2Y,point2Z; int bodyId;\n"
	"  uint pointOffset,pointCount,planeOffset,planeCount,triangleOffset,triangleCount,type,supported;\n"
	"  float friction,restitution,rollingResistance,rollingRadius,tangentVelocityX,tangentVelocityY,tangentVelocityZ,materialPadding; };\n"
	"struct ConvexManifoldResult { uint eligible,touching,pointCount,inputIndex; float nx,ny,nz; uint contactGeneration;\n"
	"  float p1x,p1y,p1z,separation1,p2x,p2y,p2z,separation2; uint feature1,feature2,scanOffset,contactId;\n"
	"  float normalImpulse1,normalImpulse2; uint persistedBits,residentFlags;\n"
	"  float friction,restitution,rollingResistance,materialPadding,tangentVelocityX,tangentVelocityY,tangentVelocityZ,tangentVelocityPadding;\n"
	"  float anchorB1X,anchorB1Y,anchorB1Z,anchorB1Padding,anchorB2X,anchorB2Y,anchorB2Z,anchorB2Padding; };\n"
	"struct ImpulsePoint { float normalImpulse,totalNormalImpulse,normalVelocity; uint featureId; };\n"
	"struct ImpulseResult { uint contactId,generation,pointCount,contactGeneration; float frictionX,frictionY,frictionZ,twistImpulse;\n"
	"  float rollingX,rollingY,rollingZ,padding; ImpulsePoint points[2]; };\n"
	"struct PreparePoint { float anchorAX,anchorAY,anchorAZ,separation; float anchorBX,anchorBY,anchorBZ,normalImpulse; uint featureId; };\n"
	"struct PrepareInput { uint contactId; int indexA,indexB; uint generation; ulong manifold; float friction,restitution;\n"
	"  float rollingResistance,tangentVelocityX,tangentVelocityY,tangentVelocityZ;\n"
	"  float twistImpulse,frictionImpulseX,frictionImpulseY,frictionImpulseZ;\n"
	"  float rollingImpulseX,rollingImpulseY,rollingImpulseZ; uint contactGeneration; PreparePoint points[2]; };\n"
	"struct ConvexManifoldParams { uint contactCount; float linearSlop,speculativeDistance; uint bodyCount; };\n"
	"struct ManifoldCompactParams { uint contactCount,blockCount,previousCount,previousGeneration,currentGeneration,p0,p1,p2; };\n"
	"struct TreeOffsets { uint offset0,offset1,offset2,padding; };\n"
	"struct TreeRefitParams { uint nodeOffset,nodeCount,targetHeight,padding; };\n"
	"struct FinalizeParams { uint bodyCount; float invTimeStep; uint transformCount; uint publishTransforms; };\n"
	"struct ShapeParams { uint shapeCount; float extra; uint useResidentBounds; uint p1; };\n"
	"struct ShapeCompactParams { uint shapeCount,blockCount,p0,p1; };\n"
	"struct FusedParams {\n"
	"  uint bodyCount; float h; float maxLinearSpeed; float maxAngularSpeed;\n"
	"  float gravityX, gravityY, gravityZ; uint integratePosition;\n"
	"};\n"
	"float4 quat_mul(float4 a, float4 b) {\n"
	"  return float4(cross(a.xyz, b.xyz) + a.w * b.xyz + b.w * a.xyz,\n"
	"                a.w * b.w - dot(a.xyz, b.xyz));\n"
	"}\n"
	"float3 inv_rotate(float4 q, float3 v) {\n"
	"  float3 t1 = cross(q.xyz, v);\n"
	"  float3 t2 = t1 - q.w * v;\n"
	"  return v + 2.0f * cross(q.xyz, t2);\n"
	"}\n"
	"float3 rotate(float4 q, float3 v) {\n"
	"  float3 t1 = cross(q.xyz, v);\n"
	"  float3 t2 = t1 + q.w * v;\n"
	"  return v + 2.0f * cross(q.xyz, t2);\n"
	"}\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"ulong b3_vf64_add_float(ulong a, float b) {\n"
	"  uint flags=0; ulong bb=soft_format_to_f64_status(ulong(as_type<uint>(b)),8u,23u,127,flags);\n"
	"  return soft_add64_status(a,bb,soft_round_near_even,flags);\n"
	"}\n"
	"float b3_vf64_bound(ulong center,float delta,float origin,float local,uint roundingMode) {\n"
	"  ulong value=b3_vf64_add_float(center,delta); value=b3_vf64_add_float(value,origin);\n"
	"  value=b3_vf64_add_float(value,local); uint flags=0;\n"
	"  return as_type<float>(uint(soft_f64_to_format_status(value,roundingMode,8u,23u,127,flags)));\n"
	"}\n"
	"float b3_vf64_difference(ulong b,ulong a) {\n"
	"  uint flags=0; ulong value=soft_sub64_status(b,a,soft_round_near_even,flags);\n"
	"  return as_type<float>(uint(soft_f64_to_format_status(value,soft_round_near_even,8u,23u,127,flags)));\n"
	"}\n"
	"#endif\n"
	"float3x3 invert_matrix(float3x3 m) {\n"
	"  float det = dot(m[0], cross(m[1], m[2]));\n"
	"  if (fabs(det) <= 1.17549435e-35f) return float3x3(0.0f);\n"
	"  float invDet = 1.0f / det;\n"
	"  float3x3 cof = float3x3(invDet * cross(m[1], m[2]),\n"
	"                           invDet * cross(m[2], m[0]),\n"
	"                           invDet * cross(m[0], m[1]));\n"
	"  return transpose(cof);\n"
	"}\n"
	"float3 solve3(float3x3 m, float3 a) {\n"
	"  float det = dot(m[0], cross(m[1], m[2]));\n"
	"  if (fabs(det) <= 1.17549435e-35f) return float3(0.0f);\n"
	"  float invDet = 1.0f / det;\n"
	"  return float3(invDet * dot(cross(m[1], m[2]), a),\n"
	"                invDet * dot(cross(m[2], m[0]), a),\n"
	"                invDet * dot(cross(m[0], m[1]), a));\n"
	"}\n"
	"kernel void b3_integrate_unconstrained(device BodyState* states [[buffer(0)]],\n"
	"                                        const device BodyProperties* props [[buffer(1)]],\n"
	"                                        constant FusedParams& p [[buffer(2)]],\n"
	"                                        uint i [[thread_position_in_grid]]) {\n"
	"  if (i >= p.bodyCount) return;\n"
	"  BodyState s = states[i];\n"
	"  BodyProperties bp = props[i];\n"
	"  float3 v = float3(s.lvx, s.lvy, s.lvz);\n"
	"  float3 w = float3(s.avx, s.avy, s.avz);\n"
	"  float linearDamping = 1.0f / (1.0f + p.h * bp.linearDamping);\n"
	"  float angularDamping = 1.0f / (1.0f + p.h * bp.angularDamping);\n"
	"  float gravityScale = bp.invMass > 0.0f ? bp.gravityScale : 0.0f;\n"
	"  float3 force = float3(bp.forceX, bp.forceY, bp.forceZ);\n"
	"  float3 gravity = float3(p.gravityX, p.gravityY, p.gravityZ);\n"
	"  v = p.h * bp.invMass * force + p.h * gravityScale * gravity + linearDamping * v;\n"
	"  float3x3 invIW = float3x3(\n"
	"    float3(bp.invInertiaWorld[0], bp.invInertiaWorld[1], bp.invInertiaWorld[2]),\n"
	"    float3(bp.invInertiaWorld[3], bp.invInertiaWorld[4], bp.invInertiaWorld[5]),\n"
	"    float3(bp.invInertiaWorld[6], bp.invInertiaWorld[7], bp.invInertiaWorld[8]));\n"
	"  float3 torque = float3(bp.torqueX, bp.torqueY, bp.torqueZ);\n"
	"  w = p.h * (invIW * torque) + angularDamping * w;\n"
	"  float4 q0 = float4(bp.qx, bp.qy, bp.qz, bp.qw);\n"
	"  float4 dq = float4(s.qx, s.qy, s.qz, s.qw);\n"
	"  float4 q = quat_mul(dq, q0);\n"
	"  float3x3 invIL = float3x3(\n"
	"    float3(bp.invInertiaLocal[0], bp.invInertiaLocal[1], bp.invInertiaLocal[2]),\n"
	"    float3(bp.invInertiaLocal[3], bp.invInertiaLocal[4], bp.invInertiaLocal[5]),\n"
	"    float3(bp.invInertiaLocal[6], bp.invInertiaLocal[7], bp.invInertiaLocal[8]));\n"
	"  float3x3 inertia = invert_matrix(invIL);\n"
	"  float3 omega1 = inv_rotate(q, w);\n"
	"  float3 omega2 = omega1;\n"
	"  float i00 = inertia[0].x, i01 = inertia[1].x, i02 = inertia[2].x;\n"
	"  float i11 = inertia[1].y, i12 = inertia[2].y, i22 = inertia[2].z;\n"
	"  float w1 = omega2.x, w2 = omega2.y, w3 = omega2.z;\n"
	"  float Iw1 = i00*w1 + i01*w2 + i02*w3;\n"
	"  float Iw2 = i01*w1 + i11*w2 + i12*w3;\n"
	"  float Iw3 = i02*w1 + i12*w2 + i22*w3;\n"
	"  float3 residual = float3(\n"
	"    p.h * (w2*Iw3 - w3*Iw2),\n"
	"    p.h * (w3*Iw1 - w1*Iw3),\n"
	"    p.h * (w1*Iw2 - w2*Iw1));\n"
	"  float3x3 J = float3x3(\n"
	"    float3(i00 + p.h*(w2*i02 - w3*i01),\n"
	"           i01 + p.h*(w3*i00 - w1*i02 - Iw3),\n"
	"           i02 + p.h*(w1*i01 - w2*i00 + Iw2)),\n"
	"    float3(i01 + p.h*(w2*i12 - w3*i11 + Iw3),\n"
	"           i11 + p.h*(w3*i01 - w1*i12),\n"
	"           i12 + p.h*(w1*i11 - w2*i01 - Iw1)),\n"
	"    float3(i02 + p.h*(w2*i22 - w3*i12 - Iw2),\n"
	"           i12 + p.h*(w3*i02 - w1*i22 + Iw1),\n"
	"           i22 + p.h*(w1*i12 - w2*i02)));\n"
	"  omega2 -= solve3(J, residual);\n"
	"  w = rotate(q, omega2);\n"
	"  if (s.flags & 0x00000001u) v.x = 0.0f;\n"
	"  if (s.flags & 0x00000002u) v.y = 0.0f;\n"
	"  if (s.flags & 0x00000004u) v.z = 0.0f;\n"
	"  if (s.flags & 0x00000008u) w.x = 0.0f;\n"
	"  if (s.flags & 0x00000010u) w.y = 0.0f;\n"
	"  if (s.flags & 0x00000020u) w.z = 0.0f;\n"
	"  float v2 = dot(v, v), maxV2 = p.maxLinearSpeed * p.maxLinearSpeed;\n"
	"  if (v2 > maxV2) { v *= p.maxLinearSpeed / sqrt(v2); s.flags |= 0x00000100u; }\n"
	"  float w2n = dot(w, w), maxW2 = p.maxAngularSpeed * p.maxAngularSpeed;\n"
	"  if (w2n > maxW2 && (s.flags & 0x00000400u) == 0u) {\n"
	"    w *= p.maxAngularSpeed / sqrt(w2n); s.flags |= 0x00000100u;\n"
	"  }\n"
	"  s.lvx = v.x; s.lvy = v.y; s.lvz = v.z;\n"
	"  s.avx = w.x; s.avy = w.y; s.avz = w.z;\n"
	"  if (p.integratePosition != 0u) {\n"
	"    float3 dp = float3(s.dpx, s.dpy, s.dpz) + p.h * v;\n"
	"    float3 qdv = 0.5f * p.h * w;\n"
	"    float3 qv = dq.xyz + cross(qdv, dq.xyz) + dq.w * qdv;\n"
	"    float qw = dq.w - dot(qdv, dq.xyz);\n"
	"    float q2 = dot(qv, qv) + qw * qw;\n"
	"    if (q2 > 1.17549435e-35f) { float invQ = 1.0f / sqrt(q2); qv *= invQ; qw *= invQ; }\n"
	"    else { qv = float3(0.0f); qw = 1.0f; }\n"
	"    s.dpx = dp.x; s.dpy = dp.y; s.dpz = dp.z;\n"
	"    s.qx = qv.x; s.qy = qv.y; s.qz = qv.z; s.qw = qw;\n"
	"  }\n"
	"  states[i] = s;\n"
	"}\n"
	"kernel void b3_integrate_positions(device BodyState* states [[buffer(0)]],\n"
	"                                   constant Params& p [[buffer(1)]],\n"
	"                                   uint i [[thread_position_in_grid]]) {\n"
	"  if (i >= p.bodyCount) return;\n"
	"  BodyState s = states[i];\n"
	"  float3 v = float3(s.lvx, s.lvy, s.lvz);\n"
	"  float3 w = float3(s.avx, s.avy, s.avz);\n"
	"  if (s.flags & 0x00000001u) v.x = 0.0f;\n"
	"  if (s.flags & 0x00000002u) v.y = 0.0f;\n"
	"  if (s.flags & 0x00000004u) v.z = 0.0f;\n"
	"  if (s.flags & 0x00000008u) w.x = 0.0f;\n"
	"  if (s.flags & 0x00000010u) w.y = 0.0f;\n"
	"  if (s.flags & 0x00000020u) w.z = 0.0f;\n"
	"  float v2 = dot(v, v);\n"
	"  float maxV2 = p.maxLinearSpeed * p.maxLinearSpeed;\n"
	"  if (v2 > maxV2) { v *= p.maxLinearSpeed / sqrt(v2); s.flags |= 0x00000100u; }\n"
	"  float w2 = dot(w, w);\n"
	"  float maxW2 = p.maxAngularSpeed * p.maxAngularSpeed;\n"
	"  if (w2 > maxW2 && (s.flags & 0x00000400u) == 0u) {\n"
	"    w *= p.maxAngularSpeed / sqrt(w2); s.flags |= 0x00000100u;\n"
	"  }\n"
	"  float3 dp = float3(s.dpx, s.dpy, s.dpz) + p.h * v;\n"
	"  float4 q = float4(s.qx, s.qy, s.qz, s.qw);\n"
	"  float3 qdv = 0.5f * p.h * w;\n"
	"  float3 qv = q.xyz + cross(qdv, q.xyz) + q.w * qdv;\n"
	"  float qw = q.w - dot(qdv, q.xyz);\n"
	"  float q2 = dot(qv, qv) + qw * qw;\n"
	"  if (q2 > 1.17549435e-35f) { float invQ = 1.0f / sqrt(q2); qv *= invQ; qw *= invQ; }\n"
	"  else { qv = float3(0.0f); qw = 1.0f; }\n"
	"  s.lvx = v.x; s.lvy = v.y; s.lvz = v.z;\n"
	"  s.avx = w.x; s.avy = w.y; s.avz = w.z;\n"
	"  s.dpx = dp.x; s.dpy = dp.y; s.dpz = dp.z;\n"
	"  s.qx = qv.x; s.qy = qv.y; s.qz = qv.z; s.qw = qw;\n"
	"  states[i] = s;\n"
	"}\n"
	"kernel void b3_finalize_bodies(device BodyState* states [[buffer(0)]],\n"
	"                               device BodyProperties* props [[buffer(1)]],\n"
	"                               const device FinalizeProperties* finalizeProps [[buffer(2)]],\n"
	"                               device FinalizeResult* results [[buffer(3)]],\n"
	"                               constant FinalizeParams& p [[buffer(4)]],\n"
	"                               device BodyTransform* bodyTransforms [[buffer(5)]],\n"
	"                               device BodyMoveResult* moveResults [[buffer(6)]],\n"
	"                               uint i [[thread_position_in_grid]]) {\n"
	"  if (i >= p.bodyCount) return;\n"
	"  BodyState s = states[i]; BodyProperties bp = props[i]; FinalizeProperties fp = finalizeProps[i];\n"
	"  float4 baseQ = float4(bp.qx, bp.qy, bp.qz, bp.qw);\n"
	"  float4 dq = float4(s.qx, s.qy, s.qz, s.qw);\n"
	"  float4 q = quat_mul(dq, baseQ);\n"
	"  float q2 = dot(q, q);\n"
	"  q = q2 > 1.17549435e-35f ? q * rsqrt(q2) : float4(0.0f, 0.0f, 0.0f, 1.0f);\n"
	"  float3 v = float3(s.lvx, s.lvy, s.lvz);\n"
	"  float3 w = float3(s.avx, s.avy, s.avz);\n"
	"  float3 localOmega = abs(inv_rotate(baseQ, w));\n"
	"  float3 localDelta = abs(inv_rotate(baseQ, dq.xyz));\n"
	"  float3 extent = float3(fp.maxExtentX, fp.maxExtentY, fp.maxExtentZ);\n"
	"  float3 velocityArc = float3(localOmega.y*extent.z + localOmega.z*extent.y,\n"
	"                              localOmega.z*extent.x + localOmega.x*extent.z,\n"
	"                              localOmega.x*extent.y + localOmega.y*extent.x);\n"
	"  float3 rotationArc = float3(localDelta.y*extent.z + localDelta.z*extent.y,\n"
	"                              localDelta.z*extent.x + localDelta.x*extent.z,\n"
	"                              localDelta.x*extent.y + localDelta.y*extent.x);\n"
	"  float maxVelocity = length(v) + length(velocityArc);\n"
	"  float3 dp = float3(s.dpx, s.dpy, s.dpz);\n"
	"  float maxDelta = length(dp) + 2.0f * length(rotationArc);\n"
	"  float sleepVelocity = max(maxVelocity, 0.5f * p.invTimeStep * maxDelta);\n"
	"  float3 localCenter = float3(fp.localCenterX, fp.localCenterY, fp.localCenterZ);\n"
	"  float3 origin = -rotate(q, localCenter);\n"
	"  float3 position = float3(fp.centerX,fp.centerY,fp.centerZ) + dp + origin;\n"
	"  float3x3 R = float3x3(rotate(q, float3(1.0f,0.0f,0.0f)),\n"
	"                          rotate(q, float3(0.0f,1.0f,0.0f)),\n"
	"                          rotate(q, float3(0.0f,0.0f,1.0f)));\n"
	"  float3x3 invIL = float3x3(\n"
	"    float3(bp.invInertiaLocal[0],bp.invInertiaLocal[1],bp.invInertiaLocal[2]),\n"
	"    float3(bp.invInertiaLocal[3],bp.invInertiaLocal[4],bp.invInertiaLocal[5]),\n"
	"    float3(bp.invInertiaLocal[6],bp.invInertiaLocal[7],bp.invInertiaLocal[8]));\n"
	"  float3x3 invIW = R * invIL * transpose(R);\n"
	"  FinalizeResult out; out.dpx=dp.x; out.dpy=dp.y; out.dpz=dp.z;\n"
	"  out.qx=q.x; out.qy=q.y; out.qz=q.z; out.qw=q.w;\n"
	"  out.originX=origin.x; out.originY=origin.y; out.originZ=origin.z;\n"
	"  out.positionX=position.x; out.positionY=position.y; out.positionZ=position.z;\n"
	"  out.sleepVelocity=sleepVelocity; out.maxVelocity=maxVelocity; out.maxDeltaPosition=maxDelta;\n"
	"  out.invInertiaWorld[0]=invIW[0].x; out.invInertiaWorld[1]=invIW[0].y; out.invInertiaWorld[2]=invIW[0].z;\n"
	"  out.invInertiaWorld[3]=invIW[1].x; out.invInertiaWorld[4]=invIW[1].y; out.invInertiaWorld[5]=invIW[1].z;\n"
	"  out.invInertiaWorld[6]=invIW[2].x; out.invInertiaWorld[7]=invIW[2].y; out.invInertiaWorld[8]=invIW[2].z;\n"
	"  results[i] = out;\n"
	"  if(p.publishTransforms!=0u&&fp.bodyId>=0&&uint(fp.bodyId)<p.transformCount){\n"
	"    BodyTransform t=bodyTransforms[fp.bodyId];t.qx=q.x;t.qy=q.y;t.qz=q.z;t.qw=q.w;\n"
	"    t.px=position.x;t.py=position.y;t.pz=position.z;t.supported=1u;t.index=int(i);t.flags=s.flags;\n"
	"    t.localCenterX=fp.localCenterX;t.localCenterY=fp.localCenterY;t.localCenterZ=fp.localCenterZ;\n"
	"    BodyMoveResult move;move.qx=q.x;move.qy=q.y;move.qz=q.z;move.qw=q.w;\n"
	"    move.px=position.x;move.py=position.y;move.pz=position.z;move.bodyId=fp.bodyId;\n"
	"    move.userData=fp.userData;move.generationWorld=fp.generationWorld;move.padding=0u;\n"
	"    bp.qx=q.x;bp.qy=q.y;bp.qz=q.z;bp.qw=q.w;\n"
	"    bp.forceX=0.0f;bp.forceY=0.0f;bp.forceZ=0.0f;bp.torqueX=0.0f;bp.torqueY=0.0f;bp.torqueZ=0.0f;\n"
	"    bp.invInertiaWorld[0]=invIW[0].x;bp.invInertiaWorld[1]=invIW[0].y;bp.invInertiaWorld[2]=invIW[0].z;\n"
	"    bp.invInertiaWorld[3]=invIW[1].x;bp.invInertiaWorld[4]=invIW[1].y;bp.invInertiaWorld[5]=invIW[1].z;\n"
	"    bp.invInertiaWorld[6]=invIW[2].x;bp.invInertiaWorld[7]=invIW[2].y;bp.invInertiaWorld[8]=invIW[2].z;props[i]=bp;\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"    t.pxBits=b3_vf64_add_float(b3_vf64_add_float(fp.centerXBits,dp.x),origin.x);\n"
	"    t.pyBits=b3_vf64_add_float(b3_vf64_add_float(fp.centerYBits,dp.y),origin.y);\n"
	"    t.pzBits=b3_vf64_add_float(b3_vf64_add_float(fp.centerZBits,dp.z),origin.z);\n"
	"    move.pxBits=t.pxBits;move.pyBits=t.pyBits;move.pzBits=t.pzBits;\n"
	"#else\n"
	"    move.pxBits=0ul;move.pyBits=0ul;move.pzBits=0ul;\n"
	"#endif\n"
	"    bodyTransforms[fp.bodyId]=t;moveResults[i]=move;\n"
	"    s.dpx=0.0f;s.dpy=0.0f;s.dpz=0.0f;s.qx=0.0f;s.qy=0.0f;s.qz=0.0f;s.qw=1.0f;\n"
	"    s.flags&=~0x00000340u;states[i]=s;\n"
	"  }\n"
	"}\n"
	"kernel void b3_finalize_shapes(const device ShapeInput* inputs [[buffer(0)]],\n"
	"                               const device FinalizeResult* bodies [[buffer(1)]],\n"
	"                               device ShapeResult* results [[buffer(2)]],\n"
	"                               constant ShapeParams& p [[buffer(3)]],\n"
	"                               const device FinalizeProperties* finalizeProps [[buffer(4)]],\n"
	"                               uint i [[thread_position_in_grid]]) {\n"
	"  if (i >= p.shapeCount) return; ShapeInput in = inputs[i]; FinalizeResult b = bodies[in.bodyIndex];\n"
	"  FinalizeProperties fp=finalizeProps[in.bodyIndex];\n"
	"  float4 q=float4(b.qx,b.qy,b.qz,b.qw); float3 localLo,localHi;\n"
	"  if (in.type == 5u) { float3 c=rotate(q,float3(in.p1x,in.p1y,in.p1z));\n"
	"    localLo=c-float3(in.radius); localHi=c+float3(in.radius);\n"
	"  } else if (in.type == 0u) {\n"
	"    float3 a=rotate(q,float3(in.p1x,in.p1y,in.p1z));\n"
	"    float3 c=rotate(q,float3(in.p2x,in.p2y,in.p2z));\n"
	"    localLo=min(a,c)-float3(in.radius); localHi=max(a,c)+float3(in.radius);\n"
	"  } else {\n"
	"    float3 sourceLo=float3(in.p1x,in.p1y,in.p1z), sourceHi=float3(in.p2x,in.p2y,in.p2z);\n"
	"    float3 center=0.5f*(sourceLo+sourceHi), extent=0.5f*(sourceHi-sourceLo);\n"
	"    float3 rx=rotate(q,float3(1,0,0)), ry=rotate(q,float3(0,1,0)), rz=rotate(q,float3(0,0,1));\n"
	"    float3 rotatedCenter=rotate(q,center); float3 worldExtent=abs(rx)*extent.x+abs(ry)*extent.y+abs(rz)*extent.z;\n"
	"    localLo=rotatedCenter-worldExtent; localHi=rotatedCenter+worldExtent;\n"
	"  }\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"  float3 arithmeticScale=max(float3(1.0f),max(abs(localLo),abs(localHi)));\n"
	"  float3 arithmeticSlack=1.9073486328125e-6f*arithmeticScale;\n"
	"  localLo-=arithmeticSlack; localHi+=arithmeticSlack;\n"
	"#endif\n"
	"  localLo-=float3(p.extra); localHi+=float3(p.extra);\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"  float3 lo=float3(b3_vf64_bound(fp.centerXBits,b.dpx,b.originX,localLo.x,soft_round_min),\n"
	"                   b3_vf64_bound(fp.centerYBits,b.dpy,b.originY,localLo.y,soft_round_min),\n"
	"                   b3_vf64_bound(fp.centerZBits,b.dpz,b.originZ,localLo.z,soft_round_min));\n"
	"  float3 hi=float3(b3_vf64_bound(fp.centerXBits,b.dpx,b.originX,localHi.x,soft_round_max),\n"
	"                   b3_vf64_bound(fp.centerYBits,b.dpy,b.originY,localHi.y,soft_round_max),\n"
	"                   b3_vf64_bound(fp.centerZBits,b.dpz,b.originZ,localHi.z,soft_round_max));\n"
	"#else\n"
	"  float3 position=float3(b.positionX,b.positionY,b.positionZ);\n"
	"  float3 lo=position+localLo, hi=position+localHi;\n"
	"#endif\n"
	"  float3 oldLo,oldHi;if(p.useResidentBounds!=0u){ShapeResult previous=results[i];\n"
	"    oldLo=float3(previous.flx,previous.fly,previous.flz);oldHi=float3(previous.fux,previous.fuy,previous.fuz);\n"
	"  }else{oldLo=float3(in.oldLx,in.oldLy,in.oldLz);oldHi=float3(in.oldUx,in.oldUy,in.oldUz);}\n"
	"  bool enlarged=any(oldLo>lo)||any(hi>oldHi); float3 fatLo=oldLo, fatHi=oldHi;\n"
	"  if (enlarged) { fatLo=lo-float3(in.margin); fatHi=hi+float3(in.margin); }\n"
	"  ShapeResult out; out.shapeId=in.shapeId; out.bodyIndex=in.bodyIndex; out.enlarged=enlarged?1u:0u; out.padding=0;\n"
	"  out.lx=lo.x;out.ly=lo.y;out.lz=lo.z;out.ux=hi.x;out.uy=hi.y;out.uz=hi.z;\n"
	"  out.flx=fatLo.x;out.fly=fatLo.y;out.flz=fatLo.z;out.fux=fatHi.x;out.fuy=fatHi.y;out.fuz=fatHi.z; results[i]=out;\n"
	"}\n"
	"kernel void b3_shape_scan_blocks(device ShapeResult* results [[buffer(0)]],device PairBlock* blocks [[buffer(1)]],\n"
	"                                 constant ShapeCompactParams& p [[buffer(2)]],uint i [[thread_position_in_grid]],\n"
	"                                 uint threadIndex [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],\n"
	"                                 ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],\n"
	"                                 ushort simdWidth [[threads_per_simdgroup]]) {\n"
	"  threadgroup uint subgroupTotals[32];threadgroup uint subgroupOffsets[32];\n"
	"  uint value=i<p.shapeCount?results[i].enlarged:0u;uint localOffset=simd_prefix_exclusive_sum(value);\n"
	"  uint subgroupTotal=simd_sum(value);if(lane==0){subgroupTotals[subgroup]=subgroupTotal;}\n"
	"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	"  if(threadIndex==0u){uint running=0u;uint subgroupCount=256u/uint(simdWidth);\n"
	"    for(uint s=0u;s<subgroupCount;++s){subgroupOffsets[s]=running;running+=subgroupTotals[s];}\n"
	"    PairBlock b;b.sum=running;b.flags=0u;b.offset=0u;b.padding=0u;blocks[group]=b;}\n"
	"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	"  if(i<p.shapeCount){ShapeResult r=results[i];r.padding=localOffset+subgroupOffsets[subgroup];results[i]=r;}\n"
	"}\n"
	"kernel void b3_shape_prefix(device PairBlock* blocks [[buffer(0)]],device PairSummary* summary [[buffer(1)]],\n"
	"                            constant ShapeCompactParams& p [[buffer(2)]]) {\n"
	"  uint total=0u;for(uint i=0u;i<p.blockCount;++i){PairBlock b=blocks[i];b.offset=total;blocks[i]=b;total+=b.sum;}\n"
	"  summary->totalCount=ulong(total);summary->flags=0u;summary->writeFlags=0u;\n"
	"}\n"
	"kernel void b3_shape_scatter(const device ShapeInput* inputs [[buffer(0)]],const device ShapeResult* results [[buffer(1)]],\n"
	"                             const device PairBlock* blocks [[buffer(2)]],device EnlargedShapeResult* compact [[buffer(3)]],\n"
	"                             constant ShapeCompactParams& p [[buffer(4)]],uint i [[thread_position_in_grid]]) {\n"
	"  if(i>=p.shapeCount)return;ShapeResult r=results[i];if(r.enlarged==0u)return;\n"
	"  uint output=blocks[i/256u].offset+r.padding;ShapeInput input=inputs[i];EnlargedShapeResult c;\n"
	"  c.shapeId=int(r.shapeId);c.proxyKey=input.proxyKey;c.lx=r.flx;c.ly=r.fly;c.lz=r.flz;\n"
	"  c.ux=r.fux;c.uy=r.fuy;c.uz=r.fuz;compact[output]=c;\n"
	"}\n"
	"kernel void b3_pair_update_leaves(const device ShapeInput* inputs [[buffer(0)]],\n"
	"                                  const device ShapeResult* results [[buffer(1)]],\n"
	"                                  device TreeNode* nodes [[buffer(2)]],\n"
	"                                  constant TreeOffsets& offsets [[buffer(3)]],uint i [[thread_position_in_grid]]) {\n"
	"  ShapeInput in=inputs[i];ShapeResult r=results[i];if(r.enlarged==0u||in.proxyKey<0)return;\n"
	"  uint type=uint(in.proxyKey)&3u;uint proxy=uint(in.proxyKey)>>2u;if(type>=3u)return;\n"
	"  uint offset=type==0u?offsets.offset0:(type==1u?offsets.offset1:offsets.offset2);\n"
	"  TreeNode n=nodes[offset+proxy];if((n.flags&4u)==0u)return;\n"
	"  n.lx=r.flx;n.ly=r.fly;n.lz=r.flz;n.ux=r.fux;n.uy=r.fuy;n.uz=r.fuz;nodes[offset+proxy]=n;\n"
	"}\n"
	"kernel void b3_pair_refit(device TreeNode* nodes [[buffer(0)]],constant TreeRefitParams& p [[buffer(1)]],\n"
	"                          uint i [[thread_position_in_grid]]) {\n"
	"  if(i>=p.nodeCount)return;uint index=p.nodeOffset+i;TreeNode n=nodes[index];\n"
	"  if((n.flags&1u)==0u||(n.flags&4u)!=0u||uint(n.height)!=p.targetHeight)return;\n"
	"  TreeNode a=nodes[p.nodeOffset+n.child1],b=nodes[p.nodeOffset+n.child2];\n"
	"  n.lx=min(a.lx,b.lx);n.ly=min(a.ly,b.ly);n.lz=min(a.lz,b.lz);\n"
	"  n.ux=max(a.ux,b.ux);n.uy=max(a.uy,b.uy);n.uz=max(a.uz,b.uz);nodes[index]=n;\n"
	"}\n"
	"inline bool tree_overlap(TreeNode n,float3 lo,float3 hi) {\n"
	"  return !(n.ux<lo.x||n.lx>hi.x||n.uy<lo.y||n.ly>hi.y||n.uz<lo.z||n.lz>hi.z);\n"
	"}\n"
	"inline bool pair_shapes_collide(PairShape a,PairShape b) {\n"
	"  if(a.bodyId==b.bodyId||a.sensorIndex>=0||b.sensorIndex>=0)return false;\n"
	"  if(a.groupIndex==b.groupIndex&&a.groupIndex!=0)return a.groupIndex>0;\n"
	"  return (a.maskBits&b.categoryBits)!=0ul&&(a.categoryBits&b.maskBits)!=0ul;\n"
	"}\n"
	"inline uint pair_key_hash(ulong key) {\n"
	"  ulong h=key;h^=h>>33;h*=0xff51afd7ed558ccdul;h^=h>>33;\n"
	"  h*=0xc4ceb9fe1a85ec53ul;h^=h>>33;return uint(h);\n"
	"}\n"
	"inline bool pair_set_contains(const device SetItem* items,uint capacity,uint shapeA,uint shapeB) {\n"
	"  uint lo=min(shapeA,shapeB),hi=max(shapeA,shapeB);\n"
	"  ulong key=(ulong(lo&0x3fffffu)<<42)|(ulong(hi&0x3fffffu)<<20);\n"
	"  uint index=pair_key_hash(key)&(capacity-1u);\n"
	"  for(uint probe=0u;probe<capacity;++probe){SetItem item=items[index];\n"
	"    if(item.hash==0u)return false;if(item.key==key)return true;index=(index+1u)&(capacity-1u);}\n"
	"  return false;\n"
	"}\n"
	"inline void query_pair_tree(const device TreeNode* nodes,const device uint* moved,const device PairShape* shapes,\n"
	"                            const device SetItem* pairSet,uint shapeCount,uint pairCapacity,uint queryShapeIndex,\n"
	"                            PairShape queryShape,int root,uint nodeOffset,int treeType,\n"
	"                            int queryKey,int queryType,float3 lo,float3 hi,\n"
	"                            device PairCandidate* candidates,uint outputOffset,uint expected,uint writeCandidates,\n"
	"                            thread int* stack,thread uint& candidateCount,thread uint& queryFlags) {\n"
	"  if(root<0||queryFlags!=0u) return; uint stackCount=0u; stack[stackCount++]=root;\n"
	"  while(stackCount>0u) { int nodeId=stack[--stackCount]; TreeNode n=nodes[nodeOffset+uint(nodeId)];\n"
	"    if(n.categoryBits==0ul||!tree_overlap(n,lo,hi)) continue;\n"
	"    if((n.flags&4u)!=0u) {\n"
	"      int targetKey=(nodeId<<2)|treeType;if(targetKey==queryKey)continue;\n"
	"      bool targetMoved=moved[nodeOffset+uint(nodeId)]!=0u;\n"
	"      if((queryType==2&&treeType==2&&targetKey<queryKey&&targetMoved)||(queryType!=2&&targetMoved))continue;\n"
	"      uint shapeIndex=n.child1;if(shapeIndex>=shapeCount||shapes[shapeIndex].bodyId<0){queryFlags|=16u;return;}\n"
	"      if(!pair_shapes_collide(queryShape,shapes[shapeIndex]))continue;\n"
	"      if(shapes[shapeIndex].type!=1u&&pair_set_contains(pairSet,pairCapacity,queryShapeIndex,shapeIndex))continue;\n"
	"      if(writeCandidates!=0u) { if(candidateCount>=expected) { queryFlags|=2u; return; }\n"
	"        PairCandidate c; c.proxyId=nodeId;c.treeType=treeType;c.shapeIndex=int(n.child1);c.padding=0;\n"
	"        candidates[outputOffset+candidateCount]=c; } candidateCount+=1u;\n"
	"    } else { if(stackCount>=63u) { queryFlags|=1u; return; }\n"
	"      stack[stackCount++]=int(n.child1); stack[stackCount++]=int(n.child2); }\n"
	"  }\n"
	"}\n"
	"kernel void b3_pair_candidates(const device int* moves [[buffer(0)]],const device TreeNode* nodes [[buffer(1)]],\n"
	"                               device PairQueryRecord* records [[buffer(2)]],device PairCandidate* candidates [[buffer(3)]],\n"
	"                               constant PairParams& p [[buffer(4)]],device PairSummary* summary [[buffer(5)]],\n"
	"                               const device uint* moved [[buffer(6)]],\n"
	"                               const device PairShape* shapes [[buffer(7)]],\n"
	"                               const device SetItem* pairSet [[buffer(8)]],\n"
	"                               uint i [[thread_position_in_grid]]) {\n"
	"  if(i>=p.moveCount) return; int key=moves[i];int queryType=key&3;int proxyId=key>>2;\n"
	"  if(p.writeCandidates!=0u&&summary->flags!=0u) return;\n"
	"  uint queryOffset=queryType==0?p.offset0:(queryType==1?p.offset1:p.offset2);\n"
	"  TreeNode q=nodes[queryOffset+uint(proxyId)];float3 lo=float3(q.lx,q.ly,q.lz),hi=float3(q.ux,q.uy,q.uz);\n"
	"  PairQueryRecord record=records[i];uint count=0u,flags=0u;thread int stack[64];\n"
	"  if(q.child1>=p.shapeCount||shapes[q.child1].bodyId<0){record.flags=16u;records[i]=record;return;}\n"
	"  PairShape queryShape=shapes[q.child1];\n"
	"  record.queryShapeIndex=int(q.child1);record.lx=q.lx;record.ly=q.ly;record.lz=q.lz;\n"
	"  record.ux=q.ux;record.uy=q.uy;record.uz=q.uz;\n"
	"  if(queryType==2) {\n"
	"    query_pair_tree(nodes,moved,shapes,pairSet,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root1,p.offset1,1,key,queryType,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags);\n"
	"    query_pair_tree(nodes,moved,shapes,pairSet,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root0,p.offset0,0,key,queryType,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags);\n"
	"  }\n"
	"  query_pair_tree(nodes,moved,shapes,pairSet,p.shapeCount,p.pairCapacity,q.child1,queryShape,p.root2,p.offset2,2,key,queryType,lo,hi,candidates,record.offset,record.count,p.writeCandidates,stack,count,flags);\n"
	"  if(p.writeCandidates==0u) record.count=count; else if(count!=record.count) flags|=2u;\n"
	"  record.flags=flags;records[i]=record;\n"
	"  if(p.writeCandidates!=0u&&flags!=0u) atomic_fetch_or_explicit((device atomic_uint*)&summary->writeFlags,8u,memory_order_relaxed);\n"
	"}\n"
	"kernel void b3_pair_scan_blocks(device PairQueryRecord* records [[buffer(0)]],device PairBlock* blocks [[buffer(1)]],\n"
	"                                constant PairPrefixParams& p [[buffer(2)]],uint i [[thread_position_in_grid]],\n"
	"                                uint threadIndex [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],\n"
	"                                ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],\n"
	"                                ushort simdWidth [[threads_per_simdgroup]]) {\n"
	"  threadgroup uint subgroupTotals[32];threadgroup uint subgroupOffsets[32];threadgroup uint subgroupErrors[32];\n"
	"  uint value=0u,error=0u;if(i<p.moveCount){PairQueryRecord r=records[i];value=r.count;error=r.flags;}\n"
	"  uint localOffset=simd_prefix_exclusive_sum(value);uint subgroupTotal=simd_sum(value);uint subgroupError=simd_max(error);\n"
	"  if(lane==0){subgroupTotals[subgroup]=subgroupTotal;subgroupErrors[subgroup]=subgroupError;}\n"
	"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	"  if(threadIndex==0u){uint running=0u,flags=0u;uint subgroupCount=256u/uint(simdWidth);\n"
	"    for(uint s=0u;s<subgroupCount;++s){subgroupOffsets[s]=running;running+=subgroupTotals[s];flags|=subgroupErrors[s];}\n"
	"    PairBlock b;b.sum=running;b.flags=flags;b.offset=0u;b.padding=0u;blocks[group]=b;}\n"
	"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
	"  if(i<p.moveCount){PairQueryRecord r=records[i];r.offset=localOffset+subgroupOffsets[subgroup];records[i]=r;}\n"
	"}\n"
	"kernel void b3_pair_prefix(device PairBlock* blocks [[buffer(0)]],device PairSummary* summary [[buffer(1)]],\n"
	"                           constant PairPrefixParams& p [[buffer(2)]]) {\n"
	"  ulong total=0ul;uint flags=0u;uint blockCount=(p.moveCount+255u)/256u;\n"
	"  for(uint i=0u;i<blockCount;++i){PairBlock b=blocks[i];if(b.flags!=0u)flags|=1u;\n"
	"    b.offset=uint(min(total,ulong(0xffffffffu)));blocks[i]=b;total+=ulong(b.sum);}\n"
	"  if(total>ulong(p.candidateLimit))flags|=2u;if(total>ulong(p.candidateCapacity))flags|=4u;\n"
	"  summary->totalCount=total;summary->flags=flags;summary->writeFlags=0u;\n"
	"}\n"
	"kernel void b3_pair_add_offsets(device PairQueryRecord* records [[buffer(0)]],const device PairBlock* blocks [[buffer(1)]],\n"
	"                                constant PairPrefixParams& p [[buffer(2)]],uint i [[thread_position_in_grid]]) {\n"
	"  if(i<p.moveCount){PairQueryRecord r=records[i];r.offset+=blocks[i/256u].offset;records[i]=r;}\n"
	"}\n"
	"float3 point_segment(float3 a,float3 b,float3 q){float3 ab=b-a;float alpha=dot(ab,q-a);\n"
	"  if(alpha<=0.0f)return a;float denominator=dot(ab,ab);if(alpha>denominator)return b;return a+(alpha/denominator)*ab;}\n"
	"struct SegmentResult { float3 point1,point2; float fraction1,fraction2; };\n"
	"SegmentResult segment_distance(float3 p1,float3 q1,float3 p2,float3 q2){SegmentResult r;float3 d1=q1-p1,d2=q2-p2,delta=p1-p2;\n"
	"  float a=dot(d1,d1),b=dot(d1,d2),c=dot(d1,delta),e=dot(d2,d2),f=dot(d2,delta);float s1,s2;\n"
	"  if(a<100.0f*1.19209290e-7f&&e<100.0f*1.19209290e-7f){s1=0.0f;s2=0.0f;}\n"
	"  else if(a<100.0f*1.19209290e-7f){s1=0.0f;s2=clamp(f/e,0.0f,1.0f);}\n"
	"  else if(e<100.0f*1.19209290e-7f){s1=clamp(-c/a,0.0f,1.0f);s2=0.0f;}\n"
	"  else{float denom=a*e-b*b;s1=denom>1000.0f*1.17549435e-38f?clamp((b*f-c*e)/denom,0.0f,1.0f):0.0f;\n"
	"    s2=(b*s1+f)/e;if(s2<0.0f){s1=clamp(-c/a,0.0f,1.0f);s2=0.0f;}else if(s2>1.0f){s1=clamp((b-c)/a,0.0f,1.0f);s2=1.0f;}}\n"
	"  r.point1=p1+s1*d1;r.fraction1=s1;r.point2=p2+s2*d2;r.fraction2=s2;return r;}\n"
	"uint clip_segment(thread float3& a,thread uint& fa,thread float3& b,thread uint& fb,float3 n,float planeOffset){\n"
	"  float da=dot(n,a)-planeOffset,db=dot(n,b)-planeOffset;float3 oa=a,ob=b;uint ofa=fa,ofb=fb;uint count=0u;\n"
	"  if(da<=0.0f){a=oa;fa=ofa;count=1u;}if(db<=0.0f){if(count==0u){a=ob;fa=ofb;}else{b=ob;fb=ofb;}count++;}\n"
	"  if(da*db<0.0f){float t=da/(da-db);float3 v=(1.0f-t)*oa+t*ob;uint feature=da>0.0f?ofa:ofb;\n"
	"    if(count==0u){a=v;fa=feature;}else{b=v;fb=feature;}count++;}return count;}\n"
	"float3 closest_triangle(float3 p,float3 a,float3 b,float3 c){float3 ab=b-a,ac=c-a,ap=p-a;float d1=dot(ab,ap),d2=dot(ac,ap);\n"
	"  if(d1<=0.0f&&d2<=0.0f)return a;float3 bp=p-b;float d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0.0f&&d4<=d3)return b;\n"
	"  float vc=d1*d4-d3*d2;if(vc<=0.0f&&d1>=0.0f&&d3<=0.0f){float v=d1/(d1-d3);return a+v*ab;}\n"
	"  float3 cp=p-c;float d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0.0f&&d5<=d6)return c;\n"
	"  float vb=d5*d2-d1*d6;if(vb<=0.0f&&d2>=0.0f&&d6<=0.0f){float w=d2/(d2-d6);return a+w*ac;}\n"
	"  float va=d3*d6-d5*d4;if(va<=0.0f&&(d4-d3)>=0.0f&&(d5-d6)>=0.0f){float w=(d4-d3)/((d4-d3)+(d5-d6));return b+w*(c-b);}\n"
	"  float denom=1.0f/(va+vb+vc),v=vb*denom,w=vc*denom;return a+v*ab+w*ac;}\n"
	"kernel void b3_convex_manifolds(const device ConvexManifoldInput* inputs [[buffer(0)]],\n"
	"                                device ConvexManifoldResult* results [[buffer(1)]],\n"
	"                                constant ConvexManifoldParams& p [[buffer(2)]],\n"
	"                                const device float4* hullPoints [[buffer(3)]],\n"
	"                                const device float4* hullPlanes [[buffer(4)]],\n"
	"                                const device HullTriangle* hullTriangles [[buffer(5)]],\n"
	"                                const device ShapeGeometry* shapeGeometry [[buffer(6)]],\n"
	"                                const device BodyTransform* bodyTransforms [[buffer(7)]],\n"
	"                                uint i [[thread_position_in_grid]]) {\n"
	"  if(i>=p.contactCount)return; ConvexManifoldInput in=inputs[i]; ConvexManifoldResult out={};out.inputIndex=i;\n"
	"  if(in.eligible==0u){results[i]=out;return;} out.eligible=1u;\n"
	"  ShapeGeometry geometryA=shapeGeometry[in.shapeIdA],geometryB=shapeGeometry[in.shapeIdB];\n"
	"  if(geometryA.supported==0u||geometryB.supported==0u||geometryA.bodyId<0||geometryB.bodyId<0||\n"
	"     uint(geometryA.bodyId)>=p.bodyCount||uint(geometryB.bodyId)>=p.bodyCount){out.eligible=0u;results[i]=out;return;}\n"
	"  BodyTransform transformA=bodyTransforms[geometryA.bodyId],transformB=bodyTransforms[geometryB.bodyId];\n"
	"  if(transformA.supported==0u||transformB.supported==0u){out.eligible=0u;results[i]=out;return;}float3 d;\n"
	"  if(((transformA.flags|transformB.flags)&0x40u)!=0u)out.residentFlags|=4u;\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"  d=float3(b3_vf64_difference(transformB.pxBits,transformA.pxBits),\n"
	"           b3_vf64_difference(transformB.pyBits,transformA.pyBits),\n"
	"           b3_vf64_difference(transformB.pzBits,transformA.pzBits));\n"
	"#else\n"
	"  d=float3(transformB.px-transformA.px,transformB.py-transformA.py,transformB.pz-transformA.pz);\n"
	"#endif\n"
	"  float4 qA=float4(transformA.qx,transformA.qy,transformA.qz,transformA.qw);\n"
	"  float4 qB=float4(transformB.qx,transformB.qy,transformB.qz,transformB.qw);\n"
	"  float3 relativePosition=inv_rotate(qA,d); float4 relativeRotation=quat_mul(float4(-qA.xyz,qA.w),qB);\n"
	"  float3 a1=float3(geometryA.point1X,geometryA.point1Y,geometryA.point1Z),a2=float3(geometryA.point2X,geometryA.point2Y,geometryA.point2Z);\n"
	"  float3 b1=rotate(relativeRotation,float3(geometryB.point1X,geometryB.point1Y,geometryB.point1Z))+relativePosition;\n"
	"  float3 b2=rotate(relativeRotation,float3(geometryB.point2X,geometryB.point2Y,geometryB.point2Z))+relativePosition;\n"
	"  if(geometryA.type==3u){ShapeGeometry hull=geometryA;\n"
	"    float bestPlane=-3.40282347e+38f;float3 bestNormal=float3(0.0f,1.0f,0.0f);uint bestFace=0u;\n"
	"    for(uint j=0u;j<hull.planeCount;++j){float4 plane=hullPlanes[hull.planeOffset+j];float separation=dot(plane.xyz,b1)-plane.w;\n"
	"      if(separation>bestPlane){bestPlane=separation;bestNormal=plane.xyz;bestFace=j;}}\n"
	"    float distance=0.0f;float3 closest=b1;float3 normal=bestNormal;\n"
	"    if(bestPlane>0.0f){float bestDistanceSquared=3.40282347e+38f;float3 projection=b1-bestPlane*bestNormal;float faceDistanceSquared=3.40282347e+38f;\n"
	"      for(uint j=0u;j<hull.triangleCount;++j){HullTriangle tri=hullTriangles[hull.triangleOffset+j];if(tri.face!=bestFace)continue;\n"
	"        float3 q=closest_triangle(projection,hullPoints[hull.pointOffset+tri.index1].xyz,hullPoints[hull.pointOffset+tri.index2].xyz,hullPoints[hull.pointOffset+tri.index3].xyz);\n"
	"        faceDistanceSquared=min(faceDistanceSquared,dot(projection-q,projection-q));}\n"
	"      float faceTolerance=0.01f*p.linearSlop;if(faceDistanceSquared<=faceTolerance*faceTolerance){closest=projection;bestDistanceSquared=bestPlane*bestPlane;}else\n"
	"      for(uint j=0u;j<hull.triangleCount;++j){HullTriangle tri=hullTriangles[hull.triangleOffset+j];\n"
	"        float3 q=closest_triangle(b1,hullPoints[hull.pointOffset+tri.index1].xyz,hullPoints[hull.pointOffset+tri.index2].xyz,hullPoints[hull.pointOffset+tri.index3].xyz);\n"
	"        float distanceSquared=dot(b1-q,b1-q);if(distanceSquared<bestDistanceSquared){bestDistanceSquared=distanceSquared;closest=q;}}\n"
	"      distance=sqrt(bestDistanceSquared);if(distance>geometryB.radius+p.speculativeDistance){results[i]=out;return;}\n"
	"      if(distance>geometryB.radius){out.eligible=0u;results[i]=out;return;}\n"
	"      if(distance*distance>1000.0f*1.17549435e-38f)normal=(b1-closest)/distance;\n"
	"    }else{distance=0.0f;closest=b1;}\n"
	"    float separation=bestPlane<=0.0f?bestPlane-geometryB.radius:distance-geometryB.radius;float3 point=0.5f*(closest+b1-geometryB.radius*normal);\n"
	"    out.touching=1u;out.pointCount=1u;out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point.x;out.p1y=point.y;out.p1z=point.z;\n"
	"    out.separation1=separation;out.feature1=0u;results[i]=out;return;}\n"
	"  float radius=geometryA.radius+geometryB.radius;float3 cpA,cpB;\n"
	"  if(geometryA.type==5u){cpA=a1;cpB=b1;}else if(geometryB.type==5u){cpB=b1;cpA=point_segment(a1,a2,cpB);}\n"
	"  else{SegmentResult sr=segment_distance(a1,a2,b1,b2);cpA=sr.point1;cpB=sr.point2;float3 initialOffset=cpB-cpA;\n"
	"    float distanceSquared=dot(initialOffset,initialOffset),maxDistance=radius+p.speculativeDistance,minDistance=0.01f*p.linearSlop;\n"
	"    if(distanceSquared>maxDistance*maxDistance||distanceSquared<minDistance*minDistance){results[i]=out;return;}\n"
	"    float3 segmentA=a2-a1,segmentB=b2-b1;float lengthA=length(segmentA),lengthB=length(segmentB);\n"
	"    if(lengthA<p.linearSlop||lengthB<p.linearSlop){results[i]=out;return;}float3 edgeA=segmentA/lengthA,edgeB=segmentB/lengthB;\n"
	"    if(dot(cross(edgeA,edgeB),cross(edgeA,edgeB))<0.0025f){float3 v1=b1,v2=b2;uint f1=0u,f2=0x00010001u;\n"
	"      uint count=clip_segment(v1,f1,v2,f2,-edgeA,-dot(edgeA,a1));if(count==2u)count=clip_segment(v1,f1,v2,f2,edgeA,dot(edgeA,a2));\n"
	"      if(count==2u){float3 c1=point_segment(a1,a2,v1),c2=point_segment(a1,a2,v2);float d1=distance(c1,v1),d2=distance(c2,v2);\n"
	"        if(d1<=radius&&d2<=radius){if(d1<minDistance||d2<minDistance){results[i]=out;return;}float3 n1=(v1-c1)/d1,n2=(v2-c2)/d2;\n"
	"          float3 normal=normalize(n1+n2);float3 point1=0.5f*((v1+geometryA.radius*n1+c1)-geometryB.radius*normal);\n"
	"          float3 point2=0.5f*((v2+geometryA.radius*n2+c2)-geometryB.radius*normal);out.touching=1u;out.pointCount=2u;\n"
	"          out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point1.x;out.p1y=point1.y;out.p1z=point1.z;out.separation1=d1-radius;\n"
	"          out.p2x=point2.x;out.p2y=point2.y;out.p2z=point2.z;out.separation2=d2-radius;out.feature1=f1;out.feature2=f2;results[i]=out;return;}}}}\n"
	"  float3 offset=cpB-cpA;float distanceSq=dot(offset,offset);if(geometryB.type==5u&&distanceSq>radius*radius){results[i]=out;return;}\n"
	"  float distance=sqrt(distanceSq);float3 normal=float3(0.0f,1.0f,0.0f);if(distance*distance>1000.0f*1.17549435e-38f)normal=offset/distance;\n"
	"  float3 point=0.5f*((cpA+geometryA.radius*normal+cpB)-geometryB.radius*normal);out.touching=1u;out.pointCount=1u;\n"
	"  out.nx=normal.x;out.ny=normal.y;out.nz=normal.z;out.p1x=point.x;out.p1y=point.y;out.p1z=point.z;\n"
	"  out.separation1=distance-radius;out.feature1=0u;results[i]=out;\n"
	"}\n"
	"inline uint b3_manifold_stable(const ConvexManifoldResult r,const ConvexManifoldInput in,const device ImpulseResult* previous,const device PrepareInput* prepareTable,constant ManifoldCompactParams& p){\n"
	"  if(p.p0==0u||r.eligible==0u||r.touching==0u||r.pointCount==0u||r.pointCount>2u||(r.residentFlags&4u)!=0u||(in.prepareEligible&2u)==0u||in.contactId>=p.previousCount)return 0u;\n"
	"  ImpulseResult old=previous[in.contactId];if(old.contactId!=in.contactId||old.generation!=p.previousGeneration||old.contactGeneration!=in.contactGeneration||old.pointCount==0u||old.pointCount>2u)return 0u;\n"
	"  PrepareInput prep=prepareTable[in.contactId];return prep.contactId==in.contactId&&prep.contactGeneration==in.contactGeneration&&prep.manifold!=0ul; }\n"
	"inline uint b3_manifold_matches(const ConvexManifoldResult r,const ConvexManifoldInput in,const device ImpulseResult* previous){ImpulseResult old=previous[in.contactId];uint claimed=0u,matches=0u;\n"
	"  for(uint pointIndex=0u;pointIndex<r.pointCount;++pointIndex){uint feature=pointIndex==0u?r.feature1:r.feature2;for(uint oldIndex=0u;oldIndex<old.pointCount;++oldIndex){uint bit=1u<<oldIndex;if((claimed&bit)==0u&&feature==old.points[oldIndex].featureId){claimed|=bit;matches+=1u;break;}}}return matches;}\n"
	"kernel void b3_manifold_scan_blocks(device ConvexManifoldResult* results [[buffer(0)]],device PairBlock* blocks [[buffer(1)]],\n"
	"  const device ConvexManifoldInput* inputs [[buffer(2)]],const device ImpulseResult* previous [[buffer(3)]],\n"
	"  const device PrepareInput* prepareTable [[buffer(4)]],constant ManifoldCompactParams& p [[buffer(5)]],uint i [[thread_position_in_grid]],uint ti [[thread_index_in_threadgroup]],\n"
	"  uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort subgroup [[simdgroup_index_in_threadgroup]],\n"
	"  ushort simdWidth [[threads_per_simdgroup]]){threadgroup uint totals[32];threadgroup uint stableTotals[32];threadgroup uint matchTotals[32];threadgroup uint offsets[32];\n"
	"  uint stable=i<p.contactCount?b3_manifold_stable(results[i],inputs[i],previous,prepareTable,p):0u;\n"
	"  uint matches=stable!=0u?b3_manifold_matches(results[i],inputs[i],previous):0u;uint value=i<p.contactCount?(p.p0!=0u?1u-stable:results[i].eligible):0u;uint local=simd_prefix_exclusive_sum(value);uint total=simd_sum(value);uint stableTotal=simd_sum(stable);uint matchTotal=simd_sum(matches);\n"
	"  if(lane==0){totals[subgroup]=total;stableTotals[subgroup]=stableTotal;matchTotals[subgroup]=matchTotal;}threadgroup_barrier(mem_flags::mem_threadgroup);if(ti==0u){uint running=0u,stableRunning=0u,matchRunning=0u;\n"
	"    for(uint s=0u;s<256u/uint(simdWidth);++s){offsets[s]=running;running+=totals[s];stableRunning+=stableTotals[s];matchRunning+=matchTotals[s];}blocks[group]=PairBlock{running,stableRunning,0u,matchRunning};}\n"
	"  threadgroup_barrier(mem_flags::mem_threadgroup);if(i<p.contactCount){ConvexManifoldResult r=results[i];\n"
	"    r.scanOffset=local+offsets[subgroup];results[i]=r;}}\n"
	"kernel void b3_manifold_prefix(device PairBlock* blocks [[buffer(0)]],device PairSummary* summary [[buffer(1)]],\n"
	"  constant ManifoldCompactParams& p [[buffer(2)]]){uint total=0u,stable=0u,matches=0u;for(uint i=0u;i<p.blockCount;++i){PairBlock b=blocks[i];\n"
	"    b.offset=total;blocks[i]=b;total+=b.sum;stable+=b.flags;matches+=b.padding;}summary->totalCount=ulong(total);summary->flags=stable;summary->writeFlags=matches;}\n"
	"kernel void b3_manifold_scatter(const device ConvexManifoldResult* results [[buffer(0)]],const device PairBlock* blocks [[buffer(1)]],\n"
	"  device ConvexManifoldResult* compact [[buffer(2)]],const device ConvexManifoldInput* inputs [[buffer(3)]],\n"
	"  const device ShapeGeometry* shapeGeometry [[buffer(4)]],const device BodyTransform* bodyTransforms [[buffer(5)]],\n"
	"  device ConvexManifoldResult* table [[buffer(6)]],const device ImpulseResult* previous [[buffer(7)]],\n"
	"  device PrepareInput* prepareTable [[buffer(8)]],constant ManifoldCompactParams& p [[buffer(9)]],uint i [[thread_position_in_grid]]){\n"
	"  if(i>=p.contactCount)return;ConvexManifoldResult r=results[i];uint contactId=inputs[i].contactId;if(r.eligible==0u){if(p.p0!=0u){uint output=blocks[i/256u].offset+r.scanOffset;r.inputIndex=i;r.scanOffset=0u;r.contactId=contactId;compact[output]=r;}return;}ShapeGeometry geometryA=shapeGeometry[inputs[i].shapeIdA];\n"
	"  ShapeGeometry geometryB=shapeGeometry[inputs[i].shapeIdB];BodyTransform transformA=bodyTransforms[geometryA.bodyId];\n"
	"  BodyTransform transformB=bodyTransforms[geometryB.bodyId];float4 q=float4(transformA.qx,transformA.qy,transformA.qz,transformA.qw);\n"
	"  float4 qb=float4(transformB.qx,transformB.qy,transformB.qz,transformB.qw);r.friction=sqrt(geometryA.friction*geometryB.friction);\n"
	"  r.restitution=max(geometryA.restitution,geometryB.restitution);r.rollingResistance=max(geometryA.rollingResistance,geometryB.rollingResistance)*max(geometryA.rollingRadius,geometryB.rollingRadius);\n"
	"  float3 tangentA=rotate(q,float3(geometryA.tangentVelocityX,geometryA.tangentVelocityY,geometryA.tangentVelocityZ));\n"
	"  float3 tangentB=rotate(qb,float3(geometryB.tangentVelocityX,geometryB.tangentVelocityY,geometryB.tangentVelocityZ));float3 tangent=tangentA-tangentB;\n"
	"  r.tangentVelocityX=tangent.x;r.tangentVelocityY=tangent.y;r.tangentVelocityZ=tangent.z;\n"
	"  float3 centerA=rotate(q,float3(transformA.localCenterX,transformA.localCenterY,transformA.localCenterZ));\n"
	"  float3 centerB=rotate(qb,float3(transformB.localCenterX,transformB.localCenterY,transformB.localCenterZ));float3 d;\n"
	"#if defined(B3_DOUBLE_PRECISION)\n"
	"  d=float3(b3_vf64_difference(transformB.pxBits,transformA.pxBits),b3_vf64_difference(transformB.pyBits,transformA.pyBits),b3_vf64_difference(transformB.pzBits,transformA.pzBits));\n"
	"#else\n"
	"  d=float3(transformB.px-transformA.px,transformB.py-transformA.py,transformB.pz-transformA.pz);\n"
	"#endif\n"
	"  if(r.pointCount>0u){float3 n=rotate(q,float3(r.nx,r.ny,r.nz));float3 p1=rotate(q,float3(r.p1x,r.p1y,r.p1z));float3 a=p1-centerA,b=p1-d-centerB;\n"
	"    r.nx=n.x;r.ny=n.y;r.nz=n.z;r.p1x=a.x;r.p1y=a.y;r.p1z=a.z;r.anchorB1X=b.x;r.anchorB1Y=b.y;r.anchorB1Z=b.z;}\n"
	"  if(r.pointCount>1u){float3 p2=rotate(q,float3(r.p2x,r.p2y,r.p2z));float3 a=p2-centerA,b=p2-d-centerB;\n"
	"    r.p2x=a.x;r.p2y=a.y;r.p2z=a.z;r.anchorB2X=b.x;r.anchorB2Y=b.y;r.anchorB2Z=b.z;}\n"
	"  r.contactGeneration=inputs[i].contactGeneration;ImpulseResult old={};uint oldValid=0u;if(r.pointCount>0u&&contactId<p.previousCount){old=previous[contactId];\n"
	"    if(old.contactId==contactId&&old.generation==p.previousGeneration&&old.contactGeneration==inputs[i].contactGeneration&&old.pointCount>0u&&old.pointCount<=2u){\n"
	"      oldValid=1u;r.residentFlags|=1u;uint claimed=0u;for(uint pointIndex=0u;pointIndex<r.pointCount;++pointIndex){uint feature=pointIndex==0u?r.feature1:r.feature2;\n"
	"        for(uint oldIndex=0u;oldIndex<old.pointCount;++oldIndex){uint bit=1u<<oldIndex;if((claimed&bit)==0u&&feature==old.points[oldIndex].featureId){\n"
	"          if(pointIndex==0u)r.normalImpulse1=old.points[oldIndex].normalImpulse;else r.normalImpulse2=old.points[oldIndex].normalImpulse;\n"
	"          r.persistedBits|=1u<<pointIndex;claimed|=bit;break;}}}}}\n"
	"  if(oldValid!=0u&&(inputs[i].prepareEligible&1u)!=0u&&r.touching!=0u&&r.pointCount>0u&&r.pointCount<=2u){PrepareInput prep=prepareTable[contactId];\n"
	"    if(prep.contactId==contactId&&prep.contactGeneration==inputs[i].contactGeneration&&prep.manifold!=0ul){prep.indexA=transformA.index;prep.indexB=transformB.index;prep.generation=p.currentGeneration;\n"
	"      prep.friction=r.friction;prep.restitution=r.restitution;prep.rollingResistance=r.rollingResistance;prep.tangentVelocityX=r.tangentVelocityX;prep.tangentVelocityY=r.tangentVelocityY;prep.tangentVelocityZ=r.tangentVelocityZ;\n"
	"      prep.twistImpulse=old.twistImpulse;prep.frictionImpulseX=old.frictionX;prep.frictionImpulseY=old.frictionY;prep.frictionImpulseZ=old.frictionZ;\n"
	"      prep.rollingImpulseX=old.rollingX;prep.rollingImpulseY=old.rollingY;prep.rollingImpulseZ=old.rollingZ;\n"
	"      prep.points[0]=PreparePoint{r.p1x,r.p1y,r.p1z,r.separation1,r.anchorB1X,r.anchorB1Y,r.anchorB1Z,r.normalImpulse1,r.feature1};\n"
	"      prep.points[1]=PreparePoint{r.p2x,r.p2y,r.p2z,r.separation2,r.anchorB2X,r.anchorB2Y,r.anchorB2Z,r.normalImpulse2,r.feature2};\n"
	"      prepareTable[contactId]=prep;r.residentFlags|=2u;}}\n"
	"  uint stable=p.p0!=0u&&(inputs[i].prepareEligible&2u)!=0u&&(r.residentFlags&6u)==2u&&r.touching!=0u&&r.pointCount>0u&&r.pointCount<=2u;if(p.p0==0u||stable==0u){uint output=blocks[i/256u].offset+r.scanOffset;r.inputIndex=i;r.scanOffset=0u;r.contactId=contactId;compact[output]=r;}\n"
	"  r.inputIndex=r.contactId;table[r.contactId]=r;}\n";
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
static const char* b3_contactSource =
	"#include <metal_stdlib>\n"
	"using namespace metal;\n"
	"struct BodyState {\n"
	"  float lvx, lvy, lvz; float avx, avy, avz; float dpx, dpy, dpz;\n"
	"  float qx, qy, qz, qw; uint flags;\n"
	"};\n"
	"struct V2W { float4 x, y; };\n"
	"struct V3W { float4 X, Y, Z; };\n"
	"struct QW { V3W V; float4 S; };\n"
	"struct Sym2W { float4 cxx, cxy, cyy; };\n"
	"struct Sym3W { float4 cxx, cxy, cxz, cyy, cyz, czz; };\n"
	"struct PointWide {\n"
	"  V3W anchorAs, anchorBs; float4 baseSeparations; float4 normalImpulses;\n"
	"  float4 totalNormalImpulses; float4 normalMasses; float4 leverArms; float4 relativeVelocities;\n"
	"};\n"
	"struct ContactWide {\n"
	"  int indexA[4]; int indexB[4]; int pointCounts[4];\n"
	"  float4 invMassA, invMassB; Sym3W invIA, invIB; V3W normal; V3W tangent1; V3W tangent2;\n"
	"  V3W centerA, centerB; float4 twistMass; float4 twistImpulse; Sym2W tangentMass;\n"
	"  V2W frictionImpulse; Sym3W rollingMass; V3W rollingImpulse; float4 friction;\n"
	"  float4 rollingResistance; float4 tangentVelocity1; float4 tangentVelocity2;\n"
	"  float4 biasRate; float4 massScale; float4 impulseScale; float4 restitution;\n"
	"  ulong manifolds[4]; PointWide points[4];\n"
	"};\n"
	"struct ConvexManifoldResult { uint eligible,touching,pointCount,inputIndex; float nx,ny,nz; uint contactGeneration;\n"
	"  float p1x,p1y,p1z,separation1,p2x,p2y,p2z,separation2; uint feature1,feature2,scanOffset,contactId;\n"
	"  float normalImpulse1,normalImpulse2; uint persistedBits,residentFlags;\n"
	"  float friction,restitution,rollingResistance,materialPadding,tangentVelocityX,tangentVelocityY,tangentVelocityZ,tangentVelocityPadding;\n"
	"  float anchorB1X,anchorB1Y,anchorB1Z,anchorB1Padding,anchorB2X,anchorB2Y,anchorB2Z,anchorB2Padding; };\n"
	"struct PreparePoint { float anchorAX,anchorAY,anchorAZ,separation; float anchorBX,anchorBY,anchorBZ,normalImpulse; uint featureId; };\n"
	"struct Softness { float biasRate, massScale, impulseScale; };\n"
	"struct PrepareInput { uint contactId; int indexA,indexB; uint generation; ulong manifold; float friction,restitution;\n"
	"  float rollingResistance,tangentVelocityX,tangentVelocityY,tangentVelocityZ;\n"
	"  float twistImpulse,frictionImpulseX,frictionImpulseY,frictionImpulseZ;\n"
	"  float rollingImpulseX,rollingImpulseY,rollingImpulseZ; uint contactGeneration; PreparePoint points[2]; };\n"
	"struct PrepareParams { uint wideCount,tableCount; float warmStartScale,invTau; Softness contactSoftness; float padding0;\n"
	"  Softness staticSoftness; uint generation; };\n"
	"struct ImpulsePoint { float normalImpulse,totalNormalImpulse,normalVelocity; uint featureId; };\n"
	"struct ImpulseResult { uint contactId,generation,pointCount,contactGeneration; float frictionX,frictionY,frictionZ,twistImpulse;\n"
	"  float rollingX,rollingY,rollingZ,padding; ImpulsePoint points[2]; };\n"
	"struct ImpulseParams { uint wideCount,tableCount,generation,padding; };\n"
	"struct BodyProperties { float qx,qy,qz,qw; float forceX,forceY,forceZ; float torqueX,torqueY,torqueZ; float invMass;\n"
	"  float invInertiaLocal[9]; float invInertiaWorld[9]; float linearDamping,angularDamping,gravityScale; };\n"
	"struct BodyW { V3W v; V3W w; V3W dp; QW dq; };\n"
	"struct ContactParams { uint offset; uint count; float invH; float contactSpeed; uint useBias; float restitutionThreshold; uint p0; uint p1; };\n"
	"struct S2 { float x, y; }; struct S3 { float x, y, z; }; struct SM2 { S2 cx, cy; }; struct SM3 { S3 cx, cy, cz; };\n"
	"struct MeshPoint { S3 rA, rB; float baseSeparation, relativeVelocity, normalImpulse, totalNormalImpulse, normalMass, leverArm; };\n"
	"struct MeshManifold { MeshPoint points[4]; int pointCount; S3 normal, tangent1, tangent2, centerA, centerB;\n"
	"  float twistMass, twistImpulse; SM2 tangentMass; S2 frictionImpulse; S3 rollingImpulse; float tangentVelocity1, tangentVelocity2; };\n"
	"struct MeshContact { ulong constraints, contact; int indexA, indexB; float invMassA, invMassB; SM3 invIA, invIB;\n"
	"  Softness softness; SM3 rollingMass; float friction, restitution, rollingResistance; int manifoldCount, manifoldStart; };\n"
	"struct BodyS { S3 v, w, dp, dqv; float dqs; };\n"
	"V3W add3(V3W a, V3W b) { return V3W{a.X+b.X,a.Y+b.Y,a.Z+b.Z}; }\n"
	"V3W sub3(V3W a, V3W b) { return V3W{a.X-b.X,a.Y-b.Y,a.Z-b.Z}; }\n"
	"V3W mul3(float4 s, V3W a) { return V3W{s*a.X,s*a.Y,s*a.Z}; }\n"
	"V3W cross3(V3W a, V3W b) {\n"
	"  return V3W{a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X};\n"
	"}\n"
	"float4 dot3(V3W a, V3W b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }\n"
	"V2W mul_sym2(Sym2W m, V2W a) { return V2W{m.cxx*a.x+m.cxy*a.y,m.cxy*a.x+m.cyy*a.y}; }\n"
	"V3W mul_sym3(Sym3W m, V3W a) {\n"
	"  return V3W{m.cxx*a.X+m.cxy*a.Y+m.cxz*a.Z,\n"
	"             m.cxy*a.X+m.cyy*a.Y+m.cyz*a.Z,\n"
	"             m.cxz*a.X+m.cyz*a.Y+m.czz*a.Z};\n"
	"}\n"
	"V3W rotate3(QW q, V3W a) {\n"
	"  V3W t1=cross3(q.V,a); V3W t2=add3(t1,mul3(q.S,a));\n"
	"  return add3(a,mul3(float4(2.0f),cross3(q.V,t2)));\n"
	"}\n"
	"BodyW gather_bodies(device BodyState* states, const device int* indices) {\n"
	"  BodyW b; b.v=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};\n"
	"  b.w=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};\n"
	"  b.dp=V3W{float4(0.0f),float4(0.0f),float4(0.0f)};\n"
	"  b.dq.V=V3W{float4(0.0f),float4(0.0f),float4(0.0f)}; b.dq.S=float4(1.0f);\n"
	"  for (uint lane=0; lane<4; ++lane) { int index=indices[lane]-1; if (index < 0) continue;\n"
	"    BodyState s=states[index]; b.v.X[lane]=s.lvx; b.v.Y[lane]=s.lvy; b.v.Z[lane]=s.lvz;\n"
	"    b.w.X[lane]=s.avx; b.w.Y[lane]=s.avy; b.w.Z[lane]=s.avz;\n"
	"    b.dp.X[lane]=s.dpx; b.dp.Y[lane]=s.dpy; b.dp.Z[lane]=s.dpz;\n"
	"    b.dq.V.X[lane]=s.qx; b.dq.V.Y[lane]=s.qy; b.dq.V.Z[lane]=s.qz; b.dq.S[lane]=s.qw;\n"
	"  } return b;\n"
	"}\n"
	"void scatter_bodies(device BodyState* states, const device int* indices, BodyW b) {\n"
	"  for (uint lane=0; lane<4; ++lane) { int index=indices[lane]-1; if (index < 0) continue;\n"
	"    BodyState s=states[index]; if ((s.flags & 0x00001000u)==0u) continue;\n"
	"    float3 v=float3(b.v.X[lane],b.v.Y[lane],b.v.Z[lane]);\n"
	"    float3 w=float3(b.w.X[lane],b.w.Y[lane],b.w.Z[lane]);\n"
	"    if (s.flags & 0x1u) v.x=0.0f; if (s.flags & 0x2u) v.y=0.0f; if (s.flags & 0x4u) v.z=0.0f;\n"
	"    if (s.flags & 0x8u) w.x=0.0f; if (s.flags & 0x10u) w.y=0.0f; if (s.flags & 0x20u) w.z=0.0f;\n"
	"    s.lvx=v.x; s.lvy=v.y; s.lvz=v.z; s.avx=w.x; s.avy=w.y; s.avz=w.z; states[index]=s;\n"
	"  }\n"
	"}\n"
	"S3 sadd(S3 a,S3 b){return S3{a.x+b.x,a.y+b.y,a.z+b.z};} S3 ssub(S3 a,S3 b){return S3{a.x-b.x,a.y-b.y,a.z-b.z};}\n"
	"S3 smul(float s,S3 a){return S3{s*a.x,s*a.y,s*a.z};} float sdot(S3 a,S3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}\n"
	"S3 scross(S3 a,S3 b){return S3{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}\n"
	"S3 smulm(SM3 m,S3 a){return S3{m.cx.x*a.x+m.cy.x*a.y+m.cz.x*a.z,m.cx.y*a.x+m.cy.y*a.y+m.cz.y*a.z,m.cx.z*a.x+m.cy.z*a.y+m.cz.z*a.z};}\n"
	"S3 sperp(S3 a){S3 p=(a.x < -0.5f || a.x > 0.5f)?S3{a.y,-a.x,0.0f}:S3{0.0f,a.z,-a.y};return smul(rsqrt(sdot(p,p)),p);}\n"
	"SM3 sinvert3(SM3 m){float det=sdot(m.cx,scross(m.cy,m.cz));if(fabs(det)<=1.17549435e-35f)return SM3{S3{0,0,0},S3{0,0,0},S3{0,0,0}};\n"
	"  float inv=1.0f/det;S3 a=smul(inv,scross(m.cy,m.cz)),b=smul(inv,scross(m.cz,m.cx)),c=smul(inv,scross(m.cx,m.cy));\n"
	"  return SM3{S3{a.x,b.x,c.x},S3{a.y,b.y,c.y},S3{a.z,b.z,c.z}};}\n"
	"SM3 load_inertia(const device BodyProperties* p,int index){if(index<0)return SM3{S3{0,0,0},S3{0,0,0},S3{0,0,0}};\n"
	"  const device float* v=p[index].invInertiaWorld;return SM3{S3{v[0],v[1],v[2]},S3{v[3],v[4],v[5]},S3{v[6],v[7],v[8]}};}\n"
	"S3 srotate(S3 qv,float qs,S3 a){S3 t1=scross(qv,a);S3 t2=sadd(t1,smul(qs,a));return sadd(a,smul(2.0f,scross(qv,t2)));}\n"
	"BodyS load_body(device BodyState* states,int index){BodyS b; b.v=S3{0,0,0};b.w=S3{0,0,0};b.dp=S3{0,0,0};b.dqv=S3{0,0,0};b.dqs=1;\n"
	"  if(index>=0){BodyState s=states[index];b.v=S3{s.lvx,s.lvy,s.lvz};b.w=S3{s.avx,s.avy,s.avz};b.dp=S3{s.dpx,s.dpy,s.dpz};b.dqv=S3{s.qx,s.qy,s.qz};b.dqs=s.qw;}return b;}\n"
	"void store_body(device BodyState* states,int index,BodyS b){if(index<0)return;BodyState s=states[index];if((s.flags&0x1000u)==0u)return;\n"
	"  if(s.flags&1u)b.v.x=0;if(s.flags&2u)b.v.y=0;if(s.flags&4u)b.v.z=0;if(s.flags&8u)b.w.x=0;if(s.flags&16u)b.w.y=0;if(s.flags&32u)b.w.z=0;\n"
	"  s.lvx=b.v.x;s.lvy=b.v.y;s.lvz=b.v.z;s.avx=b.w.x;s.avy=b.w.y;s.avz=b.w.z;states[index]=s;}\n"
	"kernel void b3_prepare_contacts(const device uint* indices [[buffer(0)]],const device PrepareInput* inputs [[buffer(1)]],\n"
	"  const device ConvexManifoldResult* table [[buffer(2)]],const device BodyProperties* properties [[buffer(3)]],\n"
	"  const device BodyState* states [[buffer(4)]],device ContactWide* constraints [[buffer(5)]],\n"
	"  device atomic_uint* status [[buffer(6)]],constant PrepareParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]){\n"
	"  if(tid>=p.wideCount)return;device ContactWide& c=constraints[tid];\n"
	"  for(uint lane=0;lane<4u;++lane){uint contactId=indices[4u*tid+lane];if(contactId==0xffffffffu)continue;\n"
	"    if(contactId>=p.tableCount){atomic_fetch_or_explicit(status,1u,memory_order_relaxed);continue;}PrepareInput in=inputs[contactId];\n"
	"    ConvexManifoldResult mr=table[contactId];if(in.contactId!=contactId||in.generation!=p.generation||mr.eligible==0u||mr.touching==0u||mr.contactId!=contactId||\n"
	"      mr.inputIndex!=contactId||mr.pointCount==0u||mr.pointCount>2u){atomic_fetch_or_explicit(status,2u,memory_order_relaxed);continue;}\n"
	"    uint pointCount=mr.pointCount;int ia=in.indexA,ib=in.indexB;c.indexA[lane]=ia+1;c.indexB[lane]=ib+1;c.pointCounts[lane]=int(pointCount);c.manifolds[lane]=in.manifold;\n"
	"    float ma=ia>=0?properties[ia].invMass:0.0f,mb=ib>=0?properties[ib].invMass:0.0f;c.invMassA[lane]=ma;c.invMassB[lane]=mb;\n"
	"    SM3 iA=load_inertia(properties,ia),iB=load_inertia(properties,ib);\n"
	"    c.invIA.cxx[lane]=iA.cx.x;c.invIA.cxy[lane]=iA.cy.x;c.invIA.cxz[lane]=iA.cz.x;c.invIA.cyy[lane]=iA.cy.y;c.invIA.cyz[lane]=iA.cz.y;c.invIA.czz[lane]=iA.cz.z;\n"
	"    c.invIB.cxx[lane]=iB.cx.x;c.invIB.cxy[lane]=iB.cy.x;c.invIB.cxz[lane]=iB.cz.x;c.invIB.cyy[lane]=iB.cy.y;c.invIB.cyz[lane]=iB.cz.y;c.invIB.czz[lane]=iB.cz.z;\n"
	"    S3 n=S3{mr.nx,mr.ny,mr.nz},t1=sperp(n),t2=scross(t1,n),tv=S3{in.tangentVelocityX,in.tangentVelocityY,in.tangentVelocityZ};\n"
	"    c.normal.X[lane]=n.x;c.normal.Y[lane]=n.y;c.normal.Z[lane]=n.z;c.tangent1.X[lane]=t1.x;c.tangent1.Y[lane]=t1.y;c.tangent1.Z[lane]=t1.z;\n"
	"    c.tangent2.X[lane]=t2.x;c.tangent2.Y[lane]=t2.y;c.tangent2.Z[lane]=t2.z;c.friction[lane]=in.friction;c.restitution[lane]=in.restitution;\n"
	"    c.rollingResistance[lane]=in.rollingResistance;c.tangentVelocity1[lane]=sdot(tv,t1);c.tangentVelocity2[lane]=sdot(tv,t2);\n"
	"    Softness soft=(ia<0||ib<0)?p.staticSoftness:p.contactSoftness;c.biasRate[lane]=soft.biasRate;c.massScale[lane]=soft.massScale;c.impulseScale[lane]=soft.impulseScale;\n"
	"    BodyState sa={};BodyState sb={};if(ia>=0)sa=states[ia];if(ib>=0)sb=states[ib];S3 va=S3{sa.lvx,sa.lvy,sa.lvz},wa=S3{sa.avx,sa.avy,sa.avz};\n"
	"    S3 vb=S3{sb.lvx,sb.lvy,sb.lvz},wb=S3{sb.avx,sb.avy,sb.avz},centerA=S3{0,0,0},centerB=S3{0,0,0};float totalWeight=0.0f;\n"
	"    for(uint j=0;j<pointCount;++j){PreparePoint mp=in.points[j];device PointWide& cp=c.points[j];S3 rA=S3{mp.anchorAX,mp.anchorAY,mp.anchorAZ},rB=S3{mp.anchorBX,mp.anchorBY,mp.anchorBZ};\n"
	"      float weight=clamp(2.0f-mp.separation*p.invTau,1.0e-10f,1.0f);centerA=sadd(centerA,smul(weight,rA));centerB=sadd(centerB,smul(weight,rB));totalWeight+=weight;\n"
	"      cp.anchorAs.X[lane]=rA.x;cp.anchorAs.Y[lane]=rA.y;cp.anchorAs.Z[lane]=rA.z;cp.anchorBs.X[lane]=rB.x;cp.anchorBs.Y[lane]=rB.y;cp.anchorBs.Z[lane]=rB.z;\n"
	"      cp.baseSeparations[lane]=mp.separation-sdot(ssub(rB,rA),n);cp.normalImpulses[lane]=p.warmStartScale*mp.normalImpulse;cp.totalNormalImpulses[lane]=0.0f;\n"
	"      S3 rnA=scross(rA,n),rnB=scross(rB,n);float k=ma+mb+sdot(rnA,smulm(iA,rnA))+sdot(rnB,smulm(iB,rnB));cp.normalMasses[lane]=k>0.0f?1.0f/k:0.0f;\n"
	"      S3 vrA=sadd(va,scross(wa,rA)),vrB=sadd(vb,scross(wb,rB));cp.relativeVelocities[lane]=sdot(n,ssub(vrB,vrA));}\n"
	"    float iw=1.0f/totalWeight;centerA=smul(iw,centerA);centerB=smul(iw,centerB);c.centerA.X[lane]=centerA.x;c.centerA.Y[lane]=centerA.y;c.centerA.Z[lane]=centerA.z;\n"
	"    c.centerB.X[lane]=centerB.x;c.centerB.Y[lane]=centerB.y;c.centerB.Z[lane]=centerB.z;\n"
	"    for(uint j=0;j<pointCount;++j){S3 rA=S3{in.points[j].anchorAX,in.points[j].anchorAY,in.points[j].anchorAZ};c.points[j].leverArms[lane]=length(float3(rA.x-centerA.x,rA.y-centerA.y,rA.z-centerA.z));}\n"
	"    for(uint j=pointCount;j<4u;++j){device PointWide& cp=c.points[j];cp.anchorAs.X[lane]=0;cp.anchorAs.Y[lane]=0;cp.anchorAs.Z[lane]=0;cp.anchorBs.X[lane]=0;cp.anchorBs.Y[lane]=0;cp.anchorBs.Z[lane]=0;\n"
	"      cp.baseSeparations[lane]=0;cp.normalImpulses[lane]=0;cp.totalNormalImpulses[lane]=0;cp.normalMasses[lane]=0;cp.leverArms[lane]=0;cp.relativeVelocities[lane]=0;}\n"
	"    S3 ra1=scross(centerA,t1),ra2=scross(centerA,t2),rb1=scross(centerB,t1),rb2=scross(centerB,t2);float kxx=ma+mb+sdot(ra1,smulm(iA,ra1))+sdot(rb1,smulm(iB,rb1));\n"
	"    float kyy=ma+mb+sdot(ra2,smulm(iA,ra2))+sdot(rb2,smulm(iB,rb2)),kxy=sdot(ra1,smulm(iA,ra2))+sdot(rb1,smulm(iB,rb2));float det=kxx*kyy-kxy*kxy;\n"
	"    float mxx=0,mxy=0,myy=0;if(fabs(det)>1.17549435e-35f){float id=1.0f/det;mxx=id*kyy;mxy=-id*kxy;myy=id*kxx;}c.tangentMass.cxx[lane]=mxx;c.tangentMass.cxy[lane]=mxy;c.tangentMass.cyy[lane]=myy;\n"
	"    S3 fi=S3{in.frictionImpulseX,in.frictionImpulseY,in.frictionImpulseZ};c.frictionImpulse.x[lane]=p.warmStartScale*sdot(fi,t1);c.frictionImpulse.y[lane]=p.warmStartScale*sdot(fi,t2);\n"
	"    SM3 sum=SM3{sadd(iA.cx,iB.cx),sadd(iA.cy,iB.cy),sadd(iA.cz,iB.cz)};float kt=sdot(n,smulm(sum,n));c.twistMass[lane]=kt>0.0f?1.0f/kt:0.0f;c.twistImpulse[lane]=p.warmStartScale*in.twistImpulse;\n"
	"    SM3 rm=sinvert3(sum);c.rollingMass.cxx[lane]=rm.cx.x;c.rollingMass.cxy[lane]=rm.cy.x;c.rollingMass.cxz[lane]=rm.cz.x;c.rollingMass.cyy[lane]=rm.cy.y;c.rollingMass.cyz[lane]=rm.cz.y;c.rollingMass.czz[lane]=rm.cz.z;\n"
	"    c.rollingImpulse.X[lane]=p.warmStartScale*in.rollingImpulseX;c.rollingImpulse.Y[lane]=p.warmStartScale*in.rollingImpulseY;c.rollingImpulse.Z[lane]=p.warmStartScale*in.rollingImpulseZ;\n"
	"  }}\n"
	"kernel void b3_warm_start_contacts(device BodyState* states [[buffer(0)]],\n"
	"                                 device ContactWide* constraints [[buffer(1)]],\n"
	"                                 constant ContactParams& p [[buffer(2)]],\n"
	"                                 uint tid [[thread_position_in_grid]]) {\n"
	"  if (tid >= p.count) return; device ContactWide& c=constraints[p.offset+tid];\n"
	"  BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);\n"
	"  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));\n"
	"  for (int j=0; j<pointCount; ++j) { device PointWide& cp=c.points[j];\n"
	"    V3W impulse=mul3(cp.normalImpulses,c.normal);\n"
	"    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));\n"
	"    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));\n"
	"  }\n"
	"  V3W frictionImpulse=add3(mul3(c.frictionImpulse.x,c.tangent1),mul3(c.frictionImpulse.y,c.tangent2));\n"
	"  a.w=sub3(a.w,mul_sym3(c.invIA,cross3(c.centerA,frictionImpulse))); a.v=sub3(a.v,mul3(c.invMassA,frictionImpulse));\n"
	"  b.w=add3(b.w,mul_sym3(c.invIB,cross3(c.centerB,frictionImpulse))); b.v=add3(b.v,mul3(c.invMassB,frictionImpulse));\n"
	"  V3W twist=mul3(c.twistImpulse,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,twist)); b.w=add3(b.w,mul_sym3(c.invIB,twist));\n"
	"  a.w=sub3(a.w,mul_sym3(c.invIA,c.rollingImpulse)); b.w=add3(b.w,mul_sym3(c.invIB,c.rollingImpulse));\n"
	"  scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);\n"
	"}\n"
	"kernel void b3_solve_contacts(device BodyState* states [[buffer(0)]],\n"
	"                            device ContactWide* constraints [[buffer(1)]],\n"
	"                            constant ContactParams& p [[buffer(2)]],\n"
	"                            uint tid [[thread_position_in_grid]]) {\n"
	"  if (tid >= p.count) return; device ContactWide& c=constraints[p.offset+tid];\n"
	"  BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);\n"
	"  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));\n"
	"  float4 biasRate=p.useBias!=0u ? c.massScale*c.biasRate : float4(0.0f);\n"
	"  float4 massScale=p.useBias!=0u ? c.massScale : float4(1.0f);\n"
	"  float4 impulseScale=p.useBias!=0u ? c.impulseScale : float4(0.0f);\n"
	"  V3W dp=sub3(b.dp,a.dp); float4 totalNormal=float4(0.0f); float4 totalTwist=float4(0.0f);\n"
	"  for (int j=0; j<pointCount; ++j) { device PointWide& cp=c.points[j];\n"
	"    V3W rsA=rotate3(a.dq,cp.anchorAs); V3W rsB=rotate3(b.dq,cp.anchorBs);\n"
	"    float4 separation=dot3(c.normal,add3(dp,sub3(rsB,rsA)))+cp.baseSeparations;\n"
	"    bool4 speculative=separation>float4(0.0f);\n"
	"    float4 bias=select(max(biasRate*separation,float4(p.contactSpeed)),separation*float4(p.invH),speculative);\n"
	"    float4 pointMassScale=select(massScale,float4(1.0f),speculative);\n"
	"    float4 pointImpulseScale=select(impulseScale,float4(0.0f),speculative);\n"
	"    V3W vrA=add3(a.v,cross3(a.w,cp.anchorAs)); V3W vrB=add3(b.v,cross3(b.w,cp.anchorBs));\n"
	"    float4 vn=dot3(sub3(vrB,vrA),c.normal);\n"
	"    float4 neg=cp.normalMasses*(pointMassScale*vn+bias)+pointImpulseScale*cp.normalImpulses;\n"
	"    float4 next=max(cp.normalImpulses-neg,float4(0.0f)); float4 delta=next-cp.normalImpulses;\n"
	"    cp.normalImpulses=next; cp.totalNormalImpulses+=next; totalNormal+=next; totalTwist+=cp.leverArms*next; V3W impulse=mul3(delta,c.normal);\n"
	"    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));\n"
	"    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));\n"
	"  }\n"
	"  if (p.useBias==0u) {\n"
	"    if (any(c.rollingResistance!=float4(0.0f))) {\n"
	"      V3W rollingDelta=mul_sym3(c.rollingMass,sub3(a.w,b.w)); V3W oldRolling=c.rollingImpulse;\n"
	"      c.rollingImpulse=add3(oldRolling,rollingDelta); float4 maxRolling=c.rollingResistance*totalNormal;\n"
	"      float4 rollingLength2=dot3(c.rollingImpulse,c.rollingImpulse); bool4 clampRolling=rollingLength2>maxRolling*maxRolling+float4(1.1920929e-7f);\n"
	"      float4 rollingScale=select(float4(1.0f),maxRolling/(sqrt(rollingLength2)+float4(1.1920929e-7f)),clampRolling);\n"
	"      rollingScale=select(float4(0.0f),rollingScale,c.rollingResistance>float4(0.0f)); c.rollingImpulse=mul3(rollingScale,c.rollingImpulse);\n"
	"      rollingDelta=sub3(c.rollingImpulse,oldRolling); a.w=sub3(a.w,mul_sym3(c.invIA,rollingDelta)); b.w=add3(b.w,mul_sym3(c.invIB,rollingDelta));\n"
	"    }\n"
	"    float4 twistSpeed=dot3(c.normal,sub3(b.w,a.w)); float4 maxTwist=c.friction*totalTwist; float4 oldTwist=c.twistImpulse;\n"
	"    c.twistImpulse=clamp(oldTwist-c.twistMass*twistSpeed,-maxTwist,maxTwist); float4 twistDelta=c.twistImpulse-oldTwist;\n"
	"    V3W angularImpulse=mul3(twistDelta,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,angularImpulse)); b.w=add3(b.w,mul_sym3(c.invIB,angularImpulse));\n"
	"    V3W vrA=add3(a.v,cross3(a.w,c.centerA)); V3W vrB=add3(b.v,cross3(b.w,c.centerB)); V3W vr=sub3(vrB,vrA);\n"
	"    V2W vt=V2W{dot3(vr,c.tangent1)-c.tangentVelocity1,dot3(vr,c.tangent2)-c.tangentVelocity2};\n"
	"    V2W frictionDelta=mul_sym2(c.tangentMass,vt); frictionDelta.x=-frictionDelta.x; frictionDelta.y=-frictionDelta.y;\n"
	"    V2W oldFriction=c.frictionImpulse; V2W nextFriction=V2W{oldFriction.x+frictionDelta.x,oldFriction.y+frictionDelta.y};\n"
	"    float4 maxFriction=c.friction*totalNormal; float4 frictionLength2=nextFriction.x*nextFriction.x+nextFriction.y*nextFriction.y;\n"
	"    bool4 clampFriction=frictionLength2>maxFriction*maxFriction; float4 frictionScale=select(float4(1.0f),maxFriction/(sqrt(frictionLength2)+float4(1.1920929e-7f)),clampFriction);\n"
	"    nextFriction.x*=frictionScale; nextFriction.y*=frictionScale; frictionDelta=V2W{nextFriction.x-oldFriction.x,nextFriction.y-oldFriction.y}; c.frictionImpulse=nextFriction;\n"
	"    V3W tangentImpulse=add3(mul3(frictionDelta.x,c.tangent1),mul3(frictionDelta.y,c.tangent2));\n"
	"    a.w=sub3(a.w,mul_sym3(c.invIA,cross3(c.centerA,tangentImpulse))); a.v=sub3(a.v,mul3(c.invMassA,tangentImpulse));\n"
	"    b.w=add3(b.w,mul_sym3(c.invIB,cross3(c.centerB,tangentImpulse))); b.v=add3(b.v,mul3(c.invMassB,tangentImpulse));\n"
	"  } scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);\n"
	"}\n"
	"kernel void b3_restitution_contacts(device BodyState* states [[buffer(0)]], device ContactWide* constraints [[buffer(1)]],\n"
	"                                    constant ContactParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {\n"
	"  if (tid>=p.count) return; device ContactWide& c=constraints[p.offset+tid]; BodyW a=gather_bodies(states,c.indexA); BodyW b=gather_bodies(states,c.indexB);\n"
	"  int pointCount=max(max(c.pointCounts[0],c.pointCounts[1]),max(c.pointCounts[2],c.pointCounts[3]));\n"
	"  for (int j=0;j<pointCount;++j) { device PointWide& cp=c.points[j];\n"
	"    bool4 apply=(c.restitution!=float4(0.0f))&&((cp.relativeVelocities+float4(p.restitutionThreshold))<=float4(0.0f))&&(cp.totalNormalImpulses!=float4(0.0f));\n"
	"    float4 mass=select(float4(0.0f),cp.normalMasses,apply); V3W vrA=add3(a.v,cross3(a.w,cp.anchorAs)); V3W vrB=add3(b.v,cross3(b.w,cp.anchorBs));\n"
	"    float4 vn=dot3(sub3(vrB,vrA),c.normal); float4 neg=mass*(vn+c.restitution*cp.relativeVelocities);\n"
	"    float4 next=max(cp.normalImpulses-neg,float4(0.0f)); float4 delta=next-cp.normalImpulses; cp.normalImpulses=next; cp.totalNormalImpulses+=delta;\n"
	"    V3W impulse=mul3(delta,c.normal); a.w=sub3(a.w,mul_sym3(c.invIA,cross3(cp.anchorAs,impulse))); a.v=sub3(a.v,mul3(c.invMassA,impulse));\n"
	"    b.w=add3(b.w,mul_sym3(c.invIB,cross3(cp.anchorBs,impulse))); b.v=add3(b.v,mul3(c.invMassB,impulse));\n"
	"  } scatter_bodies(states,c.indexA,a); scatter_bodies(states,c.indexB,b);\n"
	"}\n"
	"kernel void b3_store_contact_impulses(const device uint* indices [[buffer(0)]],const device ContactWide* constraints [[buffer(1)]],\n"
	"  device ImpulseResult* results [[buffer(2)]],const device PrepareInput* inputs [[buffer(3)]],constant ImpulseParams& p [[buffer(4)]],uint tid [[thread_position_in_grid]]){\n"
	"  if(tid>=p.wideCount)return;const device ContactWide& c=constraints[tid];\n"
	"  for(uint lane=0;lane<4u;++lane){uint contactId=indices[4u*tid+lane];if(contactId==0xffffffffu||contactId>=p.tableCount)continue;\n"
	"    uint pointCount=uint(c.pointCounts[lane]);if(pointCount==0u||pointCount>2u)continue;PrepareInput in=inputs[contactId];ImpulseResult r={};r.contactId=contactId;r.generation=p.generation;r.pointCount=pointCount;r.contactGeneration=in.contactGeneration;\n"
	"    float f1=c.frictionImpulse.x[lane],f2=c.frictionImpulse.y[lane];r.frictionX=f1*c.tangent1.X[lane]+f2*c.tangent2.X[lane];\n"
	"    r.frictionY=f1*c.tangent1.Y[lane]+f2*c.tangent2.Y[lane];r.frictionZ=f1*c.tangent1.Z[lane]+f2*c.tangent2.Z[lane];r.twistImpulse=c.twistImpulse[lane];\n"
	"    r.rollingX=c.rollingImpulse.X[lane];r.rollingY=c.rollingImpulse.Y[lane];r.rollingZ=c.rollingImpulse.Z[lane];\n"
	"    for(uint j=0;j<pointCount;++j){r.points[j].normalImpulse=c.points[j].normalImpulses[lane];r.points[j].totalNormalImpulse=c.points[j].totalNormalImpulses[lane];r.points[j].normalVelocity=c.points[j].relativeVelocities[lane];r.points[j].featureId=in.points[j].featureId;}\n"
	"    results[contactId]=r;}\n"
	"}\n"
	"void warm_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,uint index){\n"
	"  device MeshContact& c=contacts[index];BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);\n"
	"  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];\n"
	"    for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];S3 impulse=smul(cp.normalImpulse,m.normal);a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,impulse)));a.v=ssub(a.v,smul(c.invMassA,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));}\n"
	"    S3 f=sadd(smul(m.frictionImpulse.x,m.tangent1),smul(m.frictionImpulse.y,m.tangent2));a.w=ssub(a.w,smulm(c.invIA,scross(m.centerA,f)));a.v=ssub(a.v,smul(c.invMassA,f));b.w=sadd(b.w,smulm(c.invIB,scross(m.centerB,f)));b.v=sadd(b.v,smul(c.invMassB,f));\n"
	"    S3 twist=smul(m.twistImpulse,m.normal);a.w=ssub(a.w,smulm(c.invIA,twist));b.w=sadd(b.w,smulm(c.invIB,twist));a.w=ssub(a.w,smulm(c.invIA,m.rollingImpulse));b.w=sadd(b.w,smulm(c.invIB,m.rollingImpulse));\n"
	"  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);}\n"
	"kernel void b3_warm_start_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_mesh_one(s,c,m,p.offset+tid);}\n"
	"kernel void b3_warm_start_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)warm_mesh_one(s,c,m,p.offset+i);}\n"
	"void solve_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,constant ContactParams& p,uint index){\n"
	"  device MeshContact& c=contacts[index];BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);S3 dp=ssub(b.dp,a.dp);\n"
	"  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];float totalNormal=0,totalTwist=0;\n"
	"    for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];S3 ds=sadd(dp,ssub(srotate(b.dqv,b.dqs,cp.rB),srotate(a.dqv,a.dqs,cp.rA)));float s=sdot(ds,m.normal);s+=cp.baseSeparation;\n"
	"      float bias=0,massScale=1,impulseScale=0;if(s>0)bias=s*p.invH;else if(p.useBias!=0u){bias=max(c.softness.massScale*c.softness.biasRate*s,p.contactSpeed);massScale=c.softness.massScale;impulseScale=c.softness.impulseScale;}\n"
	"      S3 vrA=sadd(a.v,scross(a.w,cp.rA));S3 vrB=sadd(b.v,scross(b.w,cp.rB));float vn=sdot(ssub(vrB,vrA),m.normal);float delta=-cp.normalMass*(massScale*vn+bias)-impulseScale*cp.normalImpulse;\n"
	"      float next=max(cp.normalImpulse+delta,0.0f);delta=next-cp.normalImpulse;cp.normalImpulse=next;cp.totalNormalImpulse+=next;totalNormal+=next;totalTwist+=cp.leverArm*next;\n"
	"      S3 impulse=smul(delta,m.normal);a.v=ssub(a.v,smul(c.invMassA,impulse));a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,impulse)));}\n"
	"    if(p.useBias!=0u)continue;float twistSpeed=sdot(m.normal,ssub(b.w,a.w));float maxTwist=c.friction*totalTwist;float oldTwist=m.twistImpulse;m.twistImpulse=clamp(oldTwist-m.twistMass*twistSpeed,-maxTwist,maxTwist);\n"
	"    S3 twist=smul(m.twistImpulse-oldTwist,m.normal);a.w=ssub(a.w,smulm(c.invIA,twist));b.w=sadd(b.w,smulm(c.invIB,twist));\n"
	"    if(c.rollingResistance>0){S3 rd=smulm(c.rollingMass,ssub(a.w,b.w));S3 old=m.rollingImpulse;m.rollingImpulse=sadd(old,rd);float limit=c.rollingResistance*totalNormal;float mag2=sdot(m.rollingImpulse,m.rollingImpulse);if(mag2>limit*limit+1.1920929e-7f)m.rollingImpulse=smul(limit/sqrt(mag2),m.rollingImpulse);rd=ssub(m.rollingImpulse,old);a.w=ssub(a.w,smulm(c.invIA,rd));b.w=sadd(b.w,smulm(c.invIB,rd));}\n"
	"    S3 vrA=sadd(a.v,scross(a.w,m.centerA));S3 vrB=sadd(b.v,scross(b.w,m.centerB));S3 vr=ssub(vrB,vrA);float vx=sdot(vr,m.tangent1)-m.tangentVelocity1;float vy=sdot(vr,m.tangent2)-m.tangentVelocity2;\n"
	"    float dx=-(m.tangentMass.cx.x*vx+m.tangentMass.cy.x*vy);float dy=-(m.tangentMass.cx.y*vx+m.tangentMass.cy.y*vy);float oldX=m.frictionImpulse.x,oldY=m.frictionImpulse.y;float nx=oldX+dx,ny=oldY+dy;float limit=c.friction*totalNormal;float len2=nx*nx+ny*ny;if(len2>limit*limit){float scale=limit/sqrt(len2);nx*=scale;ny*=scale;}dx=nx-oldX;dy=ny-oldY;m.frictionImpulse=S2{nx,ny};\n"
	"    S3 impulse=sadd(smul(dx,m.tangent1),smul(dy,m.tangent2));a.v=ssub(a.v,smul(c.invMassA,impulse));a.w=ssub(a.w,smulm(c.invIA,scross(m.centerA,impulse)));b.v=sadd(b.v,smul(c.invMassB,impulse));b.w=sadd(b.w,smulm(c.invIB,scross(m.centerB,impulse)));\n"
	"  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);}\n"
	"kernel void b3_solve_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_mesh_one(s,c,m,p,p.offset+tid);}\n"
	"kernel void b3_solve_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)solve_mesh_one(s,c,m,p,p.offset+i);}\n"
	"void restitution_mesh_one(device BodyState* states,device MeshContact* contacts,device MeshManifold* manifolds,constant ContactParams& p,uint index){\n"
	"  device MeshContact& c=contacts[index];if(c.restitution==0)return;BodyS a=load_body(states,c.indexA);BodyS b=load_body(states,c.indexB);\n"
	"  for(int mi=0;mi<c.manifoldCount;++mi){device MeshManifold& m=manifolds[c.manifoldStart+mi];for(int j=0;j<m.pointCount;++j){device MeshPoint& cp=m.points[j];if(cp.relativeVelocity>-p.restitutionThreshold||cp.totalNormalImpulse==0)continue;\n"
	"    S3 vrA=sadd(a.v,scross(a.w,cp.rA));S3 vrB=sadd(b.v,scross(b.w,cp.rB));float vn=sdot(ssub(vrB,vrA),m.normal);float impulse=-cp.normalMass*(vn+c.restitution*cp.relativeVelocity);float next=max(cp.normalImpulse+impulse,0.0f);impulse=next-cp.normalImpulse;cp.normalImpulse=next;cp.totalNormalImpulse+=impulse;\n"
	"    S3 P=smul(impulse,m.normal);a.v=ssub(a.v,smul(c.invMassA,P));a.w=ssub(a.w,smulm(c.invIA,scross(cp.rA,P)));b.v=sadd(b.v,smul(c.invMassB,P));b.w=sadd(b.w,smulm(c.invIB,scross(cp.rB,P)));}\n"
	"  }store_body(states,c.indexA,a);store_body(states,c.indexB,b);\n"
	"}\n"
	"kernel void b3_restitution_mesh(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)restitution_mesh_one(s,c,m,p,p.offset+tid);}\n"
	"kernel void b3_restitution_mesh_overflow(device BodyState* s [[buffer(0)]],device MeshContact* c [[buffer(1)]],device MeshManifold* m [[buffer(2)]],constant ContactParams& p [[buffer(3)]]){for(uint i=0;i<p.count;++i)restitution_mesh_one(s,c,m,p,p.offset+i);}\n"
	"struct DistanceJoint { int indexA,indexB; float invMassA,invMassB; SM3 invIA,invIB; Softness constraintSoftness;\n"
	"  S3 anchorA,anchorB,deltaCenter; Softness distanceSoftness; float length,hertz,lowerSpringForce,upperSpringForce;\n"
	"  float minLength,maxLength,maxMotorForce,motorSpeed; float impulse,lowerImpulse,upperImpulse,motorImpulse,axialMass; uint flags; };\n"
	"struct JointParams { uint offset,count; float h,invH; uint useBias; uint p0,p1,p2; };\n"
	"void distance_apply(thread BodyS& a,thread BodyS& b,device DistanceJoint& j,S3 rA,S3 rB,S3 axis,float impulse){\n"
	"  S3 P=smul(impulse,axis);a.v=ssub(a.v,smul(j.invMassA,P));a.w=ssub(a.w,smulm(j.invIA,scross(rA,P)));\n"
	"  b.v=sadd(b.v,smul(j.invMassB,P));b.w=sadd(b.w,smulm(j.invIB,scross(rB,P)));}\n"
	"void warm_distance_one(device BodyState* states,device DistanceJoint* joints,uint index){device DistanceJoint& j=joints[index];\n"
	"  BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);S3 rA=srotate(a.dqv,a.dqs,j.anchorA),rB=srotate(b.dqv,b.dqs,j.anchorB);\n"
	"  S3 separation=sadd(j.deltaCenter,sadd(ssub(b.dp,a.dp),ssub(rB,rA)));float len=sqrt(sdot(separation,separation));S3 axis=len>0?smul(1.0f/len,separation):S3{0,0,0};\n"
	"  distance_apply(a,b,j,rA,rB,axis,j.impulse+j.lowerImpulse-j.upperImpulse+j.motorImpulse);store_body(states,j.indexA,a);store_body(states,j.indexB,b);}\n"
	"void solve_distance_one(device BodyState* states,device DistanceJoint* joints,constant JointParams& p,uint index){device DistanceJoint& j=joints[index];\n"
	"  BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);S3 rA=srotate(a.dqv,a.dqs,j.anchorA),rB=srotate(b.dqv,b.dqs,j.anchorB);\n"
	"  S3 separation=sadd(j.deltaCenter,sadd(ssub(b.dp,a.dp),ssub(rB,rA)));float len=sqrt(sdot(separation,separation));S3 axis=len>0?smul(1.0f/len,separation):S3{0,0,0};\n"
	"  bool soft=(j.flags&1u)!=0u&&(j.minLength<j.maxLength||(j.flags&2u)==0u);\n"
	"  if(soft){if(j.hertz>0){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.length;\n"
	"      float old=j.impulse;float impulse=-j.distanceSoftness.massScale*j.axialMass*(cdot+j.distanceSoftness.biasRate*C)-j.distanceSoftness.impulseScale*old;\n"
	"      j.impulse=clamp(old+impulse,j.lowerSpringForce*p.h,j.upperSpringForce*p.h);distance_apply(a,b,j,rA,rB,axis,j.impulse-old);}\n"
	"    if((j.flags&2u)!=0u){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.minLength,bias=0,ms=1,is=0;\n"
	"      if(C>0)bias=C*p.invH;else if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}\n"
	"      float impulse=-ms*j.axialMass*(cdot+bias)-is*j.lowerImpulse;float next=max(0.0f,j.lowerImpulse+impulse);impulse=next-j.lowerImpulse;j.lowerImpulse=next;distance_apply(a,b,j,rA,rB,axis,impulse);\n"
	"      vr=sadd(ssub(a.v,b.v),ssub(scross(a.w,rA),scross(b.w,rB)));cdot=sdot(axis,vr);C=j.maxLength-len;bias=0;ms=1;is=0;\n"
	"      if(C>0)bias=C*p.invH;else if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}\n"
	"      impulse=-ms*j.axialMass*(cdot+bias)-is*j.upperImpulse;next=max(0.0f,j.upperImpulse+impulse);impulse=next-j.upperImpulse;j.upperImpulse=next;distance_apply(a,b,j,rA,rB,axis,-impulse);}\n"
	"    if((j.flags&4u)!=0u){S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float impulse=j.axialMass*(j.motorSpeed-sdot(axis,vr));\n"
	"      float old=j.motorImpulse,maxImpulse=p.h*j.maxMotorForce;j.motorImpulse=clamp(old+impulse,-maxImpulse,maxImpulse);distance_apply(a,b,j,rA,rB,axis,j.motorImpulse-old);}\n"
	"  }else{S3 vr=sadd(ssub(b.v,a.v),ssub(scross(b.w,rB),scross(a.w,rA)));float cdot=sdot(axis,vr),C=len-j.length,bias=0,ms=1,is=0;\n"
	"    if(p.useBias!=0u){bias=j.constraintSoftness.biasRate*C;ms=j.constraintSoftness.massScale;is=j.constraintSoftness.impulseScale;}\n"
	"    float impulse=-ms*j.axialMass*(cdot+bias)-is*j.impulse;j.impulse+=impulse;distance_apply(a,b,j,rA,rB,axis,impulse);}\n"
	"  store_body(states,j.indexA,a);store_body(states,j.indexB,b);}\n"
	"kernel void b3_warm_start_distance(device BodyState* s [[buffer(0)]],device DistanceJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_distance_one(s,j,p.offset+tid);}\n"
	"kernel void b3_solve_distance(device BodyState* s [[buffer(0)]],device DistanceJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_distance_one(s,j,p,p.offset+tid);}\n"
	"struct SQ { S3 v; float s; };\n"
	"SQ sqmul(SQ a,SQ b){return SQ{sadd(sadd(scross(a.v,b.v),smul(a.s,b.v)),smul(b.s,a.v)),a.s*b.s-sdot(a.v,b.v)};}\n"
	"SQ sqinvmul(SQ a,SQ b){S3 nv=smul(-1.0f,a.v);return SQ{sadd(sadd(scross(nv,b.v),smul(a.s,b.v)),smul(b.s,nv)),a.s*b.s-sdot(nv,b.v)};}\n"
	"struct ParallelJoint { int indexA,indexB; SM3 invIA,invIB; Softness softness; S3 perpAxisX,perpAxisY; SQ quatA,quatB; float maxTorque; S2 perpImpulse; uint fixedRotation; };\n"
	"void warm_parallel_one(device BodyState* states,device ParallelJoint* joints,uint index){device ParallelJoint& j=joints[index];BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);\n"
	"  S3 impulse=sadd(smul(j.perpImpulse.x,j.perpAxisX),smul(j.perpImpulse.y,j.perpAxisY));a.w=ssub(a.w,smulm(j.invIA,impulse));b.w=sadd(b.w,smulm(j.invIB,impulse));store_body(states,j.indexA,a);store_body(states,j.indexB,b);}\n"
	"void solve_parallel_one(device BodyState* states,device ParallelJoint* joints,constant JointParams& p,uint index){device ParallelJoint& j=joints[index];BodyS a=load_body(states,j.indexA),b=load_body(states,j.indexB);\n"
	"  SQ qa=sqmul(SQ{a.dqv,a.dqs},j.quatA),qb=sqmul(SQ{b.dqv,b.dqs},j.quatB);if(sdot(qa.v,qb.v)+qa.s*qb.s<0){qb.v=smul(-1.0f,qb.v);qb.s=-qb.s;}SQ rel=sqinvmul(qa,qb);\n"
	"  if(j.fixedRotation==0u&&j.maxTorque>0){S3 ax=smul(0.5f,srotate(qa.v,qa.s,sadd(smul(rel.s,S3{1,0,0}),scross(rel.v,S3{1,0,0}))));S3 ay=smul(0.5f,srotate(qa.v,qa.s,sadd(smul(rel.s,S3{0,1,0}),scross(rel.v,S3{0,1,0}))));j.perpAxisX=ax;j.perpAxisY=ay;\n"
	"    SM3 sum=SM3{sadd(j.invIA.cx,j.invIB.cx),sadd(j.invIA.cy,j.invIB.cy),sadd(j.invIA.cz,j.invIB.cz)};float kxx=sdot(ax,smulm(sum,ax)),kyy=sdot(ay,smulm(sum,ay)),kxy=sdot(ax,smulm(sum,ay));\n"
	"    S3 wr=ssub(b.w,a.w);float bx=sdot(wr,ax)+j.softness.biasRate*rel.v.x,by=sdot(wr,ay)+j.softness.biasRate*rel.v.y;float det=kxx*kyy-kxy*kxy;float sx=0,sy=0;if(fabs(det)>1.17549435e-35f){sx=(kyy*bx-kxy*by)/det;sy=(kxx*by-kxy*bx)/det;}\n"
	"    S2 old=j.perpImpulse;S2 next=S2{old.x-j.softness.massScale*sx-j.softness.impulseScale*old.x,old.y-j.softness.massScale*sy-j.softness.impulseScale*old.y};float maxImpulse=p.h*j.maxTorque,len2=next.x*next.x+next.y*next.y;if(len2>maxImpulse*maxImpulse){float scale=maxImpulse/sqrt(len2);next.x*=scale;next.y*=scale;}j.perpImpulse=next;\n"
	"    S3 impulse=sadd(smul(next.x-old.x,ax),smul(next.y-old.y,ay));a.w=ssub(a.w,smulm(j.invIA,impulse));b.w=sadd(b.w,smulm(j.invIB,impulse));}\n"
	"  store_body(states,j.indexA,a);store_body(states,j.indexB,b);}\n"
	"kernel void b3_warm_start_parallel(device BodyState* s [[buffer(0)]],device ParallelJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)warm_parallel_one(s,j,p.offset+tid);}\n"
	"kernel void b3_solve_parallel(device BodyState* s [[buffer(0)]],device ParallelJoint* j [[buffer(1)]],constant JointParams& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){if(tid<p.count)solve_parallel_one(s,j,p,p.offset+tid);}\n"
	"struct JointOverflow { uint type,index; };\n"
	"kernel void b3_warm_start_joint_overflow(device BodyState* s [[buffer(0)]],device DistanceJoint* d [[buffer(1)]],device ParallelJoint* r [[buffer(2)]],device const JointOverflow* o [[buffer(3)]],constant JointParams& p [[buffer(4)]]){for(uint i=0;i<p.count;++i){JointOverflow e=o[i];if(e.type==0u)warm_distance_one(s,d,e.index);else warm_parallel_one(s,r,e.index);}}\n"
	"kernel void b3_solve_joint_overflow(device BodyState* s [[buffer(0)]],device DistanceJoint* d [[buffer(1)]],device ParallelJoint* r [[buffer(2)]],device const JointOverflow* o [[buffer(3)]],constant JointParams& p [[buffer(4)]]){for(uint i=0;i<p.count;++i){JointOverflow e=o[i];if(e.type==0u)solve_distance_one(s,d,p,e.index);else solve_parallel_one(s,r,p,e.index);}}\n";
#pragma clang diagnostic pop

static void b3MetalWriteError( char* buffer, int capacity, NSString* message )
{
	if ( buffer == NULL || capacity <= 0 )
	{
		return;
	}

	const char* text = message != nil ? message.UTF8String : "unknown Metal error";
	snprintf( buffer, (size_t)capacity, "%s", text );
}

bool b3MetalCreateContext( b3MetalContext** contextOut, char* errorBuffer, int errorCapacity )
{
	if ( contextOut == NULL )
	{
		b3MetalWriteError( errorBuffer, errorCapacity, @"contextOut is null" );
		return false;
	}

	*contextOut = NULL;
	@autoreleasepool
	{
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if ( device == nil )
		{
			b3MetalWriteError( errorBuffer, errorCapacity, @"no Metal device is available" );
			return false;
		}

		id<MTLCommandQueue> queue = [device newCommandQueue];
		if ( queue == nil )
		{
			[device release];
			b3MetalWriteError( errorBuffer, errorCapacity, @"failed to create Metal command queue" );
			return false;
		}

		MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
		if ( @available( macOS 15.0, iOS 18.0, * ) )
		{
			options.mathMode = MTLMathModeSafe;
		}
		else
		{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			options.fastMathEnabled = NO;
#pragma clang diagnostic pop
		}
		NSError* error = nil;
#if defined( BOX3D_DOUBLE_PRECISION )
		NSMutableString* source = [NSMutableString stringWithString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
		[source appendString:[NSString stringWithUTF8String:b3_vf64IEEESource]];
		[source appendString:@"\n#define B3_DOUBLE_PRECISION 1\n"];
		[source appendString:[NSString stringWithUTF8String:b3_metalSource]];
#else
		NSString* source = [NSString stringWithUTF8String:b3_metalSource];
#endif
		id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
		if ( library == nil )
		{
			[options release];
			[queue release];
			[device release];
			b3MetalWriteError( errorBuffer, errorCapacity, error.localizedDescription );
			return false;
		}

		id<MTLFunction> positionFunction = [library newFunctionWithName:@"b3_integrate_positions"];
		id<MTLComputePipelineState> positionPipeline =
			positionFunction != nil ? [device newComputePipelineStateWithFunction:positionFunction error:&error] : nil;
		[positionFunction release];
		id<MTLFunction> fusedFunction = [library newFunctionWithName:@"b3_integrate_unconstrained"];
		id<MTLComputePipelineState> fusedPipeline =
			fusedFunction != nil ? [device newComputePipelineStateWithFunction:fusedFunction error:&error] : nil;
		[fusedFunction release];
		id<MTLFunction> finalizeFunction = [library newFunctionWithName:@"b3_finalize_bodies"];
		id<MTLComputePipelineState> finalizePipeline =
			finalizeFunction != nil ? [device newComputePipelineStateWithFunction:finalizeFunction error:&error] : nil;
		[finalizeFunction release];
		id<MTLFunction> finalizeShapesFunction = [library newFunctionWithName:@"b3_finalize_shapes"];
		id<MTLComputePipelineState> finalizeShapesPipeline = finalizeShapesFunction != nil
			? [device newComputePipelineStateWithFunction:finalizeShapesFunction error:&error]
			: nil;
		[finalizeShapesFunction release];
		id<MTLFunction> shapeScanBlocksFunction = [library newFunctionWithName:@"b3_shape_scan_blocks"];
		id<MTLComputePipelineState> shapeScanBlocksPipeline = shapeScanBlocksFunction != nil
			? [device newComputePipelineStateWithFunction:shapeScanBlocksFunction error:&error]
			: nil;
		[shapeScanBlocksFunction release];
		id<MTLFunction> shapePrefixFunction = [library newFunctionWithName:@"b3_shape_prefix"];
		id<MTLComputePipelineState> shapePrefixPipeline = shapePrefixFunction != nil
			? [device newComputePipelineStateWithFunction:shapePrefixFunction error:&error]
			: nil;
		[shapePrefixFunction release];
		id<MTLFunction> shapeScatterFunction = [library newFunctionWithName:@"b3_shape_scatter"];
		id<MTLComputePipelineState> shapeScatterPipeline = shapeScatterFunction != nil
			? [device newComputePipelineStateWithFunction:shapeScatterFunction error:&error]
			: nil;
		[shapeScatterFunction release];
		id<MTLFunction> pairCandidatesFunction = [library newFunctionWithName:@"b3_pair_candidates"];
		id<MTLComputePipelineState> pairCandidatesPipeline = pairCandidatesFunction != nil
			? [device newComputePipelineStateWithFunction:pairCandidatesFunction error:&error]
			: nil;
		[pairCandidatesFunction release];
		id<MTLFunction> pairScanBlocksFunction = [library newFunctionWithName:@"b3_pair_scan_blocks"];
		id<MTLComputePipelineState> pairScanBlocksPipeline = pairScanBlocksFunction != nil
			? [device newComputePipelineStateWithFunction:pairScanBlocksFunction error:&error]
			: nil;
		[pairScanBlocksFunction release];
		id<MTLFunction> pairPrefixFunction = [library newFunctionWithName:@"b3_pair_prefix"];
		id<MTLComputePipelineState> pairPrefixPipeline = pairPrefixFunction != nil
			? [device newComputePipelineStateWithFunction:pairPrefixFunction error:&error]
			: nil;
		[pairPrefixFunction release];
		id<MTLFunction> pairAddOffsetsFunction = [library newFunctionWithName:@"b3_pair_add_offsets"];
		id<MTLComputePipelineState> pairAddOffsetsPipeline = pairAddOffsetsFunction != nil
			? [device newComputePipelineStateWithFunction:pairAddOffsetsFunction error:&error]
			: nil;
		[pairAddOffsetsFunction release];
		id<MTLFunction> pairUpdateLeavesFunction = [library newFunctionWithName:@"b3_pair_update_leaves"];
		id<MTLComputePipelineState> pairUpdateLeavesPipeline = pairUpdateLeavesFunction != nil
			? [device newComputePipelineStateWithFunction:pairUpdateLeavesFunction error:&error]
			: nil;
		[pairUpdateLeavesFunction release];
		id<MTLFunction> pairRefitFunction = [library newFunctionWithName:@"b3_pair_refit"];
		id<MTLComputePipelineState> pairRefitPipeline = pairRefitFunction != nil
			? [device newComputePipelineStateWithFunction:pairRefitFunction error:&error]
			: nil;
		[pairRefitFunction release];
		id<MTLFunction> convexManifoldFunction = [library newFunctionWithName:@"b3_convex_manifolds"];
		id<MTLComputePipelineState> convexManifoldPipeline = convexManifoldFunction != nil
			? [device newComputePipelineStateWithFunction:convexManifoldFunction error:&error]
			: nil;
		[convexManifoldFunction release];
		id<MTLFunction> convexManifoldScanFunction = [library newFunctionWithName:@"b3_manifold_scan_blocks"];
		id<MTLComputePipelineState> convexManifoldScanPipeline = convexManifoldScanFunction != nil
			? [device newComputePipelineStateWithFunction:convexManifoldScanFunction error:&error]
			: nil;
		[convexManifoldScanFunction release];
		id<MTLFunction> convexManifoldPrefixFunction = [library newFunctionWithName:@"b3_manifold_prefix"];
		id<MTLComputePipelineState> convexManifoldPrefixPipeline = convexManifoldPrefixFunction != nil
			? [device newComputePipelineStateWithFunction:convexManifoldPrefixFunction error:&error]
			: nil;
		[convexManifoldPrefixFunction release];
		id<MTLFunction> convexManifoldScatterFunction = [library newFunctionWithName:@"b3_manifold_scatter"];
		id<MTLComputePipelineState> convexManifoldScatterPipeline = convexManifoldScatterFunction != nil
			? [device newComputePipelineStateWithFunction:convexManifoldScatterFunction error:&error]
			: nil;
		[convexManifoldScatterFunction release];
		[library release];

		NSString* contactSource = [NSString stringWithUTF8String:b3_contactSource];
		id<MTLLibrary> contactLibrary = [device newLibraryWithSource:contactSource options:options error:&error];
		[options release];
		id<MTLFunction> prepareContactsFunction = [contactLibrary newFunctionWithName:@"b3_prepare_contacts"];
		id<MTLComputePipelineState> prepareContactsPipeline = prepareContactsFunction != nil
			? [device newComputePipelineStateWithFunction:prepareContactsFunction error:&error]
			: nil;
		[prepareContactsFunction release];
		id<MTLFunction> storeContactImpulsesFunction = [contactLibrary newFunctionWithName:@"b3_store_contact_impulses"];
		id<MTLComputePipelineState> storeContactImpulsesPipeline = storeContactImpulsesFunction != nil
			? [device newComputePipelineStateWithFunction:storeContactImpulsesFunction error:&error]
			: nil;
		[storeContactImpulsesFunction release];
		id<MTLFunction> warmStartFunction = [contactLibrary newFunctionWithName:@"b3_warm_start_contacts"];
		id<MTLComputePipelineState> warmStartPipeline =
			warmStartFunction != nil ? [device newComputePipelineStateWithFunction:warmStartFunction error:&error] : nil;
		[warmStartFunction release];
		id<MTLFunction> solveFunction = [contactLibrary newFunctionWithName:@"b3_solve_contacts"];
		id<MTLComputePipelineState> solvePipeline =
			solveFunction != nil ? [device newComputePipelineStateWithFunction:solveFunction error:&error] : nil;
		[solveFunction release];
		id<MTLFunction> restitutionFunction = [contactLibrary newFunctionWithName:@"b3_restitution_contacts"];
		id<MTLComputePipelineState> restitutionPipeline =
			restitutionFunction != nil ? [device newComputePipelineStateWithFunction:restitutionFunction error:&error] : nil;
		[restitutionFunction release];
		id<MTLFunction> warmStartMeshFunction = [contactLibrary newFunctionWithName:@"b3_warm_start_mesh"];
		id<MTLComputePipelineState> warmStartMeshPipeline =
			warmStartMeshFunction != nil ? [device newComputePipelineStateWithFunction:warmStartMeshFunction error:&error] : nil;
		[warmStartMeshFunction release];
		id<MTLFunction> solveMeshFunction = [contactLibrary newFunctionWithName:@"b3_solve_mesh"];
		id<MTLComputePipelineState> solveMeshPipeline =
			solveMeshFunction != nil ? [device newComputePipelineStateWithFunction:solveMeshFunction error:&error] : nil;
		[solveMeshFunction release];
		id<MTLFunction> restitutionMeshFunction = [contactLibrary newFunctionWithName:@"b3_restitution_mesh"];
		id<MTLComputePipelineState> restitutionMeshPipeline = restitutionMeshFunction != nil
			? [device newComputePipelineStateWithFunction:restitutionMeshFunction error:&error]
			: nil;
		[restitutionMeshFunction release];
		id<MTLFunction> warmStartOverflowFunction = [contactLibrary newFunctionWithName:@"b3_warm_start_mesh_overflow"];
		id<MTLComputePipelineState> warmStartOverflowPipeline = warmStartOverflowFunction != nil
			? [device newComputePipelineStateWithFunction:warmStartOverflowFunction error:&error]
			: nil;
		[warmStartOverflowFunction release];
		id<MTLFunction> solveOverflowFunction = [contactLibrary newFunctionWithName:@"b3_solve_mesh_overflow"];
		id<MTLComputePipelineState> solveOverflowPipeline = solveOverflowFunction != nil
			? [device newComputePipelineStateWithFunction:solveOverflowFunction error:&error]
			: nil;
		[solveOverflowFunction release];
		id<MTLFunction> restitutionOverflowFunction = [contactLibrary newFunctionWithName:@"b3_restitution_mesh_overflow"];
		id<MTLComputePipelineState> restitutionOverflowPipeline = restitutionOverflowFunction != nil
			? [device newComputePipelineStateWithFunction:restitutionOverflowFunction error:&error]
			: nil;
		[restitutionOverflowFunction release];
		id<MTLFunction> warmStartDistanceFunction = [contactLibrary newFunctionWithName:@"b3_warm_start_distance"];
		id<MTLComputePipelineState> warmStartDistancePipeline = warmStartDistanceFunction != nil
			? [device newComputePipelineStateWithFunction:warmStartDistanceFunction error:&error]
			: nil;
		[warmStartDistanceFunction release];
		id<MTLFunction> solveDistanceFunction = [contactLibrary newFunctionWithName:@"b3_solve_distance"];
		id<MTLComputePipelineState> solveDistancePipeline = solveDistanceFunction != nil
			? [device newComputePipelineStateWithFunction:solveDistanceFunction error:&error]
			: nil;
		[solveDistanceFunction release];
		id<MTLFunction> warmStartParallelFunction = [contactLibrary newFunctionWithName:@"b3_warm_start_parallel"];
		id<MTLComputePipelineState> warmStartParallelPipeline = warmStartParallelFunction != nil
			? [device newComputePipelineStateWithFunction:warmStartParallelFunction error:&error]
			: nil;
		[warmStartParallelFunction release];
		id<MTLFunction> solveParallelFunction = [contactLibrary newFunctionWithName:@"b3_solve_parallel"];
		id<MTLComputePipelineState> solveParallelPipeline = solveParallelFunction != nil
			? [device newComputePipelineStateWithFunction:solveParallelFunction error:&error]
			: nil;
		[solveParallelFunction release];
		id<MTLFunction> warmStartJointOverflowFunction =
			[contactLibrary newFunctionWithName:@"b3_warm_start_joint_overflow"];
		id<MTLComputePipelineState> warmStartJointOverflowPipeline = warmStartJointOverflowFunction != nil
			? [device newComputePipelineStateWithFunction:warmStartJointOverflowFunction error:&error]
			: nil;
		[warmStartJointOverflowFunction release];
		id<MTLFunction> solveJointOverflowFunction =
			[contactLibrary newFunctionWithName:@"b3_solve_joint_overflow"];
		id<MTLComputePipelineState> solveJointOverflowPipeline = solveJointOverflowFunction != nil
			? [device newComputePipelineStateWithFunction:solveJointOverflowFunction error:&error]
			: nil;
		[solveJointOverflowFunction release];
		[contactLibrary release];
		if ( positionPipeline == nil || fusedPipeline == nil || finalizePipeline == nil || finalizeShapesPipeline == nil ||
			 shapeScanBlocksPipeline == nil || shapePrefixPipeline == nil || shapeScatterPipeline == nil ||
			 pairCandidatesPipeline == nil || pairScanBlocksPipeline == nil || pairPrefixPipeline == nil ||
			 pairAddOffsetsPipeline == nil || pairUpdateLeavesPipeline == nil || pairRefitPipeline == nil ||
			 convexManifoldPipeline == nil || convexManifoldScanPipeline == nil || convexManifoldPrefixPipeline == nil ||
			 convexManifoldScatterPipeline == nil ||
			 prepareContactsPipeline == nil || storeContactImpulsesPipeline == nil || warmStartPipeline == nil || solvePipeline == nil ||
			 restitutionPipeline == nil || warmStartMeshPipeline == nil || solveMeshPipeline == nil || restitutionMeshPipeline == nil ||
			 warmStartOverflowPipeline == nil || solveOverflowPipeline == nil || restitutionOverflowPipeline == nil ||
			 warmStartDistancePipeline == nil || solveDistancePipeline == nil || warmStartParallelPipeline == nil ||
			 solveParallelPipeline == nil ||
			 warmStartJointOverflowPipeline == nil || solveJointOverflowPipeline == nil )
		{
			[positionPipeline release];
			[fusedPipeline release];
			[finalizePipeline release];
			[finalizeShapesPipeline release];
			[shapeScanBlocksPipeline release];
			[shapePrefixPipeline release];
			[shapeScatterPipeline release];
			[pairCandidatesPipeline release];
			[pairScanBlocksPipeline release];
			[pairPrefixPipeline release];
			[pairAddOffsetsPipeline release];
			[pairUpdateLeavesPipeline release];
			[pairRefitPipeline release];
			[convexManifoldPipeline release];
			[convexManifoldScanPipeline release];
			[convexManifoldPrefixPipeline release];
			[convexManifoldScatterPipeline release];
			[prepareContactsPipeline release];
			[storeContactImpulsesPipeline release];
			[warmStartPipeline release];
			[solvePipeline release];
			[restitutionPipeline release];
			[warmStartMeshPipeline release];
			[solveMeshPipeline release];
			[restitutionMeshPipeline release];
			[warmStartOverflowPipeline release];
			[solveOverflowPipeline release];
			[restitutionOverflowPipeline release];
			[warmStartDistancePipeline release];
			[solveDistancePipeline release];
			[warmStartParallelPipeline release];
			[solveParallelPipeline release];
			[warmStartJointOverflowPipeline release];
			[solveJointOverflowPipeline release];
			[queue release];
			[device release];
			b3MetalWriteError( errorBuffer, errorCapacity, error.localizedDescription );
			return false;
		}

		b3MetalContext* context = calloc( 1, sizeof( b3MetalContext ) );
		if ( context == NULL )
		{
			[positionPipeline release];
			[fusedPipeline release];
			[finalizePipeline release];
			[finalizeShapesPipeline release];
			[shapeScanBlocksPipeline release];
			[shapePrefixPipeline release];
			[shapeScatterPipeline release];
			[pairCandidatesPipeline release];
			[pairScanBlocksPipeline release];
			[pairPrefixPipeline release];
			[pairAddOffsetsPipeline release];
			[pairUpdateLeavesPipeline release];
			[pairRefitPipeline release];
			[convexManifoldPipeline release];
			[convexManifoldScanPipeline release];
			[convexManifoldPrefixPipeline release];
			[convexManifoldScatterPipeline release];
			[prepareContactsPipeline release];
			[storeContactImpulsesPipeline release];
			[warmStartPipeline release];
			[solvePipeline release];
			[restitutionPipeline release];
			[warmStartMeshPipeline release];
			[solveMeshPipeline release];
			[restitutionMeshPipeline release];
			[warmStartOverflowPipeline release];
			[solveOverflowPipeline release];
			[restitutionOverflowPipeline release];
			[warmStartDistancePipeline release];
			[solveDistancePipeline release];
			[warmStartParallelPipeline release];
			[solveParallelPipeline release];
			[warmStartJointOverflowPipeline release];
			[solveJointOverflowPipeline release];
			[queue release];
			[device release];
			b3MetalWriteError( errorBuffer, errorCapacity, @"failed to allocate Metal context" );
			return false;
		}

		context->device = device;
		context->queue = queue;
		context->integratePositionsPipeline = positionPipeline;
		context->integrateUnconstrainedPipeline = fusedPipeline;
		context->finalizeBodiesPipeline = finalizePipeline;
		context->finalizeShapesPipeline = finalizeShapesPipeline;
		context->shapeScanBlocksPipeline = shapeScanBlocksPipeline;
		context->shapePrefixPipeline = shapePrefixPipeline;
		context->shapeScatterPipeline = shapeScatterPipeline;
		context->pairCandidatesPipeline = pairCandidatesPipeline;
		context->pairScanBlocksPipeline = pairScanBlocksPipeline;
		context->pairPrefixPipeline = pairPrefixPipeline;
		context->pairAddOffsetsPipeline = pairAddOffsetsPipeline;
		context->pairUpdateLeavesPipeline = pairUpdateLeavesPipeline;
		context->pairRefitPipeline = pairRefitPipeline;
		context->convexManifoldPipeline = convexManifoldPipeline;
		context->convexManifoldScanPipeline = convexManifoldScanPipeline;
		context->convexManifoldPrefixPipeline = convexManifoldPrefixPipeline;
		context->convexManifoldScatterPipeline = convexManifoldScatterPipeline;
		context->convexShapeGeometryRevision = UINT64_MAX;
		context->convexShapeMaterialRevision = UINT64_MAX;
		context->convexBodyTransformStepIndex = UINT64_MAX;
		context->convexBodyTransformRevision = UINT64_MAX;
		context->prepareContactsPipeline = prepareContactsPipeline;
		context->storeContactImpulsesPipeline = storeContactImpulsesPipeline;
		context->warmStartContactsPipeline = warmStartPipeline;
		context->solveContactsPipeline = solvePipeline;
		context->restitutionContactsPipeline = restitutionPipeline;
		context->warmStartMeshPipeline = warmStartMeshPipeline;
		context->solveMeshPipeline = solveMeshPipeline;
		context->restitutionMeshPipeline = restitutionMeshPipeline;
		context->warmStartOverflowPipeline = warmStartOverflowPipeline;
		context->solveOverflowPipeline = solveOverflowPipeline;
		context->restitutionOverflowPipeline = restitutionOverflowPipeline;
		context->warmStartDistancePipeline = warmStartDistancePipeline;
		context->solveDistancePipeline = solveDistancePipeline;
		context->warmStartParallelPipeline = warmStartParallelPipeline;
		context->solveParallelPipeline = solveParallelPipeline;
		context->warmStartJointOverflowPipeline = warmStartJointOverflowPipeline;
		context->solveJointOverflowPipeline = solveJointOverflowPipeline;
		*contextOut = context;
		return true;
	}
}

void b3MetalDestroyContext( b3MetalContext* context )
{
	if ( context == NULL )
	{
		return;
	}

	[context->bodyStateBuffer release];
	[context->bodyPropertiesBuffer release];
	[context->finalizeResultBuffer release];
	[context->finalizeReadbackBuffer release];
	[context->finalizePropertiesBuffer release];
	[context->bodyMoveResultBuffer release];
	[context->bodyMoveReadbackBuffer release];
	[context->shapeInputBuffer release];
	free( context->shapeInputBodyIds );
	[context->shapeResultBuffer release];
	[context->shapeReadbackBuffer release];
	[context->shapeCompactBuffer release];
	[context->shapeBlockBuffer release];
	[context->shapeSummaryBuffer release];
	[context->pairMoveBuffer release];
	[context->pairTreeBuffer release];
	[context->pairMovedBuffer release];
	[context->pairShapeBuffer release];
	[context->pairSetBuffer release];
	[context->pairRecordBuffer release];
	[context->pairCandidateBuffer release];
	[context->pairSummaryBuffer release];
	[context->pairBlockBuffer release];
	[context->convexManifoldInputBuffer release];
	[context->convexManifoldResultBuffer release];
	[context->convexManifoldCompactBuffer release];
	[context->convexManifoldBlockBuffer release];
	[context->convexManifoldSummaryBuffer release];
	[context->convexManifoldTableBuffer release];
	[context->convexManifoldTableReadbackBuffer release];
	[context->convexHullPointBuffer release];
	[context->convexHullPlaneBuffer release];
	[context->convexHullTriangleBuffer release];
	[context->convexShapeGeometryBuffer release];
	[context->convexBodyTransformBuffer release];
	[context->contactConstraintBuffer release];
	[context->contactPrepareTableBuffer release];
	[context->contactPrepareIndexBuffer release];
	[context->contactPrepareStatusBuffer release];
	[context->contactImpulseResultBuffer release];
	[context->contactHitEventIdBuffer release];
	[context->meshContactBuffer release];
	[context->meshManifoldBuffer release];
	[context->distanceJointBuffer release];
	[context->parallelJointBuffer release];
	[context->jointOverflowBuffer release];
	[context->integratePositionsPipeline release];
	[context->integrateUnconstrainedPipeline release];
	[context->finalizeBodiesPipeline release];
	[context->finalizeShapesPipeline release];
	[context->shapeScanBlocksPipeline release];
	[context->shapePrefixPipeline release];
	[context->shapeScatterPipeline release];
	[context->pairCandidatesPipeline release];
	[context->pairScanBlocksPipeline release];
	[context->pairPrefixPipeline release];
	[context->pairAddOffsetsPipeline release];
	[context->pairUpdateLeavesPipeline release];
	[context->pairRefitPipeline release];
	[context->convexManifoldPipeline release];
	[context->convexManifoldScanPipeline release];
	[context->convexManifoldPrefixPipeline release];
	[context->convexManifoldScatterPipeline release];
	[context->prepareContactsPipeline release];
	[context->storeContactImpulsesPipeline release];
	[context->warmStartContactsPipeline release];
	[context->solveContactsPipeline release];
	[context->restitutionContactsPipeline release];
	[context->warmStartMeshPipeline release];
	[context->solveMeshPipeline release];
	[context->restitutionMeshPipeline release];
	[context->warmStartOverflowPipeline release];
	[context->solveOverflowPipeline release];
	[context->restitutionOverflowPipeline release];
	[context->warmStartDistancePipeline release];
	[context->solveDistancePipeline release];
	[context->warmStartParallelPipeline release];
	[context->solveParallelPipeline release];
	[context->warmStartJointOverflowPipeline release];
	[context->solveJointOverflowPipeline release];
	[context->queue release];
	[context->device release];
	free( context );
}

void b3MetalGetDeviceName( const b3MetalContext* context, char* nameBuffer, int nameCapacity )
{
	if ( nameBuffer == NULL || nameCapacity <= 0 )
	{
		return;
	}

	NSString* name = context != NULL ? context->device.name : @"unavailable";
	snprintf( nameBuffer, (size_t)nameCapacity, "%s", name.UTF8String );
}

static bool b3MetalEnsureBodyCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->bodyStateCapacity >= requiredBytes )
	{
		return true;
	}

	NSUInteger capacity = context->bodyStateCapacity > 0 ? context->bodyStateCapacity : 4096;
	while ( capacity < requiredBytes )
	{
		capacity *= 2;
	}

	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil )
	{
		return false;
	}

	[context->bodyStateBuffer release];
	context->bodyStateBuffer = buffer;
	context->bodyStateCapacity = capacity;
	return true;
}

static bool b3MetalEnsurePropertiesCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->bodyPropertiesCapacity >= requiredBytes )
	{
		return true;
	}

	NSUInteger capacity = context->bodyPropertiesCapacity > 0 ? context->bodyPropertiesCapacity : 4096;
	while ( capacity < requiredBytes )
	{
		capacity *= 2;
	}

	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil )
	{
		return false;
	}

	[context->bodyPropertiesBuffer release];
	context->bodyPropertiesBuffer = buffer;
	context->bodyPropertiesCapacity = capacity;
	context->bodyPropertiesResidentCount = 0;
	return true;
}

static bool b3MetalEnsureFinalizeResultCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->finalizeResultCapacity >= requiredBytes && context->finalizeReadbackCapacity >= requiredBytes )
	{
		return true;
	}

	NSUInteger capacity = context->finalizeResultCapacity > 0 ? context->finalizeResultCapacity : 4096;
	while ( capacity < requiredBytes )
	{
		capacity *= 2;
	}

	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
	id<MTLBuffer> readback = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil || readback == nil )
	{
		[buffer release];
		[readback release];
		return false;
	}

	[context->finalizeResultBuffer release];
	[context->finalizeReadbackBuffer release];
	context->finalizeResultBuffer = buffer;
	context->finalizeResultCapacity = capacity;
	context->finalizeReadbackBuffer = readback;
	context->finalizeReadbackCapacity = capacity;
	return true;
}

static bool b3MetalEnsureFinalizePropertiesCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->finalizePropertiesCapacity >= requiredBytes )
	{
		return true;
	}
	NSUInteger capacity = context->finalizePropertiesCapacity > 0 ? context->finalizePropertiesCapacity : 4096;
	while ( capacity < requiredBytes )
	{
		capacity *= 2;
	}
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil )
	{
		return false;
	}
	[context->finalizePropertiesBuffer release];
	context->finalizePropertiesBuffer = buffer;
	context->finalizePropertiesCapacity = capacity;
	return true;
}

static bool b3MetalEnsureBodyMoveCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->bodyMoveResultCapacity >= requiredBytes && context->bodyMoveReadbackCapacity >= requiredBytes )
	{
		return true;
	}
	NSUInteger capacity = context->bodyMoveResultCapacity > 0 ? context->bodyMoveResultCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> result = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
	id<MTLBuffer> readback = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( result == nil || readback == nil )
	{
		[result release];
		[readback release];
		return false;
	}
	[context->bodyMoveResultBuffer release];
	[context->bodyMoveReadbackBuffer release];
	context->bodyMoveResultBuffer = result;
	context->bodyMoveResultCapacity = capacity;
	context->bodyMoveReadbackBuffer = readback;
	context->bodyMoveReadbackCapacity = capacity;
	context->bodyMoveResultCount = 0;
	return true;
}

static bool b3MetalEnsureConvexManifoldCapacity( b3MetalContext* context, NSUInteger inputBytes, NSUInteger resultBytes,
	NSUInteger compactBytes, NSUInteger blockBytes, NSUInteger tableBytes )
{
	if ( context->convexManifoldSummaryBuffer == nil )
	{
		context->convexManifoldSummaryBuffer =
			[context->device newBufferWithLength:sizeof( b3MetalPairSummary ) options:MTLResourceStorageModeShared];
		if ( context->convexManifoldSummaryBuffer == nil ) return false;
	}
	if ( context->convexManifoldInputCapacity < inputBytes )
	{
		NSUInteger capacity = context->convexManifoldInputCapacity > 0 ? context->convexManifoldInputCapacity : 4096;
		while ( capacity < inputBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexManifoldInputBuffer release];
		context->convexManifoldInputBuffer = buffer;
		context->convexManifoldInputCapacity = capacity;
	}
	if ( context->convexManifoldResultCapacity < resultBytes )
	{
		NSUInteger capacity = context->convexManifoldResultCapacity > 0 ? context->convexManifoldResultCapacity : 4096;
		while ( capacity < resultBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->convexManifoldResultBuffer release];
		context->convexManifoldResultBuffer = buffer;
		context->convexManifoldResultCapacity = capacity;
	}
	if ( context->convexManifoldCompactCapacity < compactBytes )
	{
		NSUInteger capacity = context->convexManifoldCompactCapacity > 0 ? context->convexManifoldCompactCapacity : 4096;
		while ( capacity < compactBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexManifoldCompactBuffer release];
		context->convexManifoldCompactBuffer = buffer;
		context->convexManifoldCompactCapacity = capacity;
	}
	if ( context->convexManifoldBlockCapacity < blockBytes )
	{
		NSUInteger capacity = context->convexManifoldBlockCapacity > 0 ? context->convexManifoldBlockCapacity : 4096;
		while ( capacity < blockBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->convexManifoldBlockBuffer release];
		context->convexManifoldBlockBuffer = buffer;
		context->convexManifoldBlockCapacity = capacity;
	}
	if ( context->convexManifoldTableCapacity < tableBytes )
	{
		NSUInteger capacity = context->convexManifoldTableCapacity > 0 ? context->convexManifoldTableCapacity : 4096;
		while ( capacity < tableBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->convexManifoldTableBuffer release];
		context->convexManifoldTableBuffer = buffer;
		context->convexManifoldTableCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsureShapeGeometryCapacity( b3MetalContext* context, int recordCount, NSUInteger hullPointBytes,
	NSUInteger hullPlaneBytes, NSUInteger hullTriangleBytes )
{
	if ( recordCount < 0 || (NSUInteger)recordCount > NSUIntegerMax / sizeof( b3MetalShapeGeometry ) ) return false;
	NSUInteger recordBytes = (NSUInteger)recordCount * sizeof( b3MetalShapeGeometry );
	recordBytes = recordBytes > sizeof( b3MetalShapeGeometry ) ? recordBytes : sizeof( b3MetalShapeGeometry );
	if ( context->convexShapeGeometryCapacity < recordBytes )
	{
		NSUInteger capacity = context->convexShapeGeometryCapacity > 0 ? context->convexShapeGeometryCapacity : 4096;
		while ( capacity < recordBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexShapeGeometryBuffer release];
		context->convexShapeGeometryBuffer = buffer;
		context->convexShapeGeometryCapacity = capacity;
	}
	hullPointBytes = hullPointBytes > sizeof( b3MetalFloat4 ) ? hullPointBytes : sizeof( b3MetalFloat4 );
	hullPlaneBytes = hullPlaneBytes > sizeof( b3MetalFloat4 ) ? hullPlaneBytes : sizeof( b3MetalFloat4 );
	hullTriangleBytes = hullTriangleBytes > sizeof( b3MetalHullTriangle ) ? hullTriangleBytes : sizeof( b3MetalHullTriangle );
	if ( context->convexHullPointCapacity < hullPointBytes )
	{
		NSUInteger capacity = context->convexHullPointCapacity > 0 ? context->convexHullPointCapacity : 4096;
		while ( capacity < hullPointBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexHullPointBuffer release];
		context->convexHullPointBuffer = buffer;
		context->convexHullPointCapacity = capacity;
	}
	if ( context->convexHullPlaneCapacity < hullPlaneBytes )
	{
		NSUInteger capacity = context->convexHullPlaneCapacity > 0 ? context->convexHullPlaneCapacity : 4096;
		while ( capacity < hullPlaneBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexHullPlaneBuffer release];
		context->convexHullPlaneBuffer = buffer;
		context->convexHullPlaneCapacity = capacity;
	}
	if ( context->convexHullTriangleCapacity < hullTriangleBytes )
	{
		NSUInteger capacity = context->convexHullTriangleCapacity > 0 ? context->convexHullTriangleCapacity : 4096;
		while ( capacity < hullTriangleBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexHullTriangleBuffer release];
		context->convexHullTriangleBuffer = buffer;
		context->convexHullTriangleCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsureShapeCapacity( b3MetalContext* context, NSUInteger inputBytes, NSUInteger resultBytes )
{
	if ( context->shapeSummaryBuffer == nil )
	{
		context->shapeSummaryBuffer =
			[context->device newBufferWithLength:sizeof( b3MetalPairSummary ) options:MTLResourceStorageModeShared];
		if ( context->shapeSummaryBuffer == nil ) return false;
	}
	if ( context->shapeInputCapacity < inputBytes )
	{
		NSUInteger capacity = context->shapeInputCapacity > 0 ? context->shapeInputCapacity : 4096;
		while ( capacity < inputBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->shapeInputBuffer release];
		context->shapeInputBuffer = buffer;
		context->shapeInputCapacity = capacity;
	}
	if ( context->shapeResultCapacity < resultBytes || context->shapeReadbackCapacity < resultBytes )
	{
		NSUInteger capacity = context->shapeResultCapacity > 0 ? context->shapeResultCapacity : 4096;
		while ( capacity < resultBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		id<MTLBuffer> readback = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil || readback == nil )
		{
			[buffer release];
			[readback release];
			return false;
		}
		[context->shapeResultBuffer release];
		[context->shapeReadbackBuffer release];
		context->shapeResultBuffer = buffer;
		context->shapeResultCapacity = capacity;
		context->shapeReadbackBuffer = readback;
		context->shapeReadbackCapacity = capacity;
		context->shapeBoundsRevision = 0;
		context->shapeBoundsCount = 0;
	}
	NSUInteger shapeCount = resultBytes / sizeof( b3MetalShapeAABBResult );
	NSUInteger compactBytes = shapeCount * sizeof( b3MetalEnlargedShapeResult );
	if ( context->shapeCompactCapacity < compactBytes )
	{
		NSUInteger capacity = context->shapeCompactCapacity > 0 ? context->shapeCompactCapacity : 4096;
		while ( capacity < compactBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->shapeCompactBuffer release];
		context->shapeCompactBuffer = buffer;
		context->shapeCompactCapacity = capacity;
	}
	NSUInteger blockCount = ( shapeCount + 255 ) / 256;
	NSUInteger blockBytes = blockCount * sizeof( b3MetalPairBlock );
	if ( context->shapeBlockCapacity < blockBytes )
	{
		NSUInteger capacity = context->shapeBlockCapacity > 0 ? context->shapeBlockCapacity : 4096;
		while ( capacity < blockBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->shapeBlockBuffer release];
		context->shapeBlockBuffer = buffer;
		context->shapeBlockCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsurePairCapacity( b3MetalContext* context, NSUInteger moveBytes, NSUInteger treeBytes,
	NSUInteger movedBytes, NSUInteger shapeBytes, NSUInteger setBytes, NSUInteger recordBytes, NSUInteger candidateBytes,
	NSUInteger blockBytes )
{
	if ( context->pairSummaryBuffer == nil )
	{
		context->pairSummaryBuffer =
			[context->device newBufferWithLength:sizeof( b3MetalPairSummary ) options:MTLResourceStorageModeShared];
		if ( context->pairSummaryBuffer == nil ) return false;
	}
	if ( context->pairMoveCapacity < moveBytes )
	{
		NSUInteger capacity = context->pairMoveCapacity > 0 ? context->pairMoveCapacity : 4096;
		while ( capacity < moveBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairMoveBuffer release];
		context->pairMoveBuffer = buffer;
		context->pairMoveCapacity = capacity;
	}
	if ( context->pairTreeCapacity < treeBytes )
	{
		NSUInteger capacity = context->pairTreeCapacity > 0 ? context->pairTreeCapacity : 4096;
		while ( capacity < treeBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairTreeBuffer release];
		context->pairTreeBuffer = buffer;
		context->pairTreeCapacity = capacity;
		context->pairTreeRevision = 0;
	}
	if ( context->pairMovedCapacity < movedBytes )
	{
		NSUInteger capacity = context->pairMovedCapacity > 0 ? context->pairMovedCapacity : 4096;
		while ( capacity < movedBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairMovedBuffer release];
		context->pairMovedBuffer = buffer;
		context->pairMovedCapacity = capacity;
	}
	if ( context->pairShapeCapacity < shapeBytes )
	{
		NSUInteger capacity = context->pairShapeCapacity > 0 ? context->pairShapeCapacity : 4096;
		while ( capacity < shapeBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairShapeBuffer release];
		context->pairShapeBuffer = buffer;
		context->pairShapeCapacity = capacity;
		context->pairShapeRevision = UINT64_MAX;
	}
	if ( context->pairSetCapacity < setBytes )
	{
		NSUInteger capacity = context->pairSetCapacity > 0 ? context->pairSetCapacity : 4096;
		while ( capacity < setBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairSetBuffer release];
		context->pairSetBuffer = buffer;
		context->pairSetCapacity = capacity;
		context->pairSetRevision = UINT64_MAX;
	}
	if ( context->pairRecordCapacity < recordBytes )
	{
		NSUInteger capacity = context->pairRecordCapacity > 0 ? context->pairRecordCapacity : 4096;
		while ( capacity < recordBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairRecordBuffer release];
		context->pairRecordBuffer = buffer;
		context->pairRecordCapacity = capacity;
	}
	if ( context->pairBlockCapacity < blockBytes )
	{
		NSUInteger capacity = context->pairBlockCapacity > 0 ? context->pairBlockCapacity : 4096;
		while ( capacity < blockBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairBlockBuffer release];
		context->pairBlockBuffer = buffer;
		context->pairBlockCapacity = capacity;
	}
	candidateBytes = candidateBytes > 0 ? candidateBytes : sizeof( b3MetalPairCandidate );
	if ( context->pairCandidateCapacity < candidateBytes )
	{
		NSUInteger capacity = context->pairCandidateCapacity > 0 ? context->pairCandidateCapacity : 4096;
		while ( capacity < candidateBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairCandidateBuffer release];
		context->pairCandidateBuffer = buffer;
		context->pairCandidateCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsureContactCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->contactConstraintCapacity >= requiredBytes )
	{
		return true;
	}

	NSUInteger capacity = context->contactConstraintCapacity > 0 ? context->contactConstraintCapacity : 4096;
	while ( capacity < requiredBytes )
	{
		capacity *= 2;
	}

	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil )
	{
		return false;
	}

	[context->contactConstraintBuffer release];
	context->contactConstraintBuffer = buffer;
	context->contactConstraintCapacity = capacity;
	return true;
}

static bool b3MetalEnsureContactPrepareTableCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->contactPrepareTableCapacity < requiredBytes )
	{
		NSUInteger capacity = context->contactPrepareTableCapacity > 0 ? context->contactPrepareTableCapacity : 4096;
		while ( capacity < requiredBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		memset( buffer.contents, 0, capacity );
		[context->contactPrepareTableBuffer release];
		context->contactPrepareTableBuffer = buffer;
		context->contactPrepareTableCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsureContactPrepareIndexCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->contactPrepareIndexCapacity < requiredBytes )
	{
		NSUInteger capacity = context->contactPrepareIndexCapacity > 0 ? context->contactPrepareIndexCapacity : 4096;
		while ( capacity < requiredBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->contactPrepareIndexBuffer release];
		context->contactPrepareIndexBuffer = buffer;
		context->contactPrepareIndexCapacity = capacity;
	}
	if ( context->contactPrepareStatusBuffer == nil )
	{
		context->contactPrepareStatusBuffer =
			[context->device newBufferWithLength:sizeof( uint32_t ) options:MTLResourceStorageModeShared];
		if ( context->contactPrepareStatusBuffer == nil ) return false;
	}
	return true;
}

static bool b3MetalEnsureContactImpulseResultCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->contactImpulseResultCapacity < requiredBytes )
	{
		NSUInteger capacity = context->contactImpulseResultCapacity > 0 ? context->contactImpulseResultCapacity : 4096;
		while ( capacity < requiredBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		memset( buffer.contents, 0, capacity );
		[context->contactImpulseResultBuffer release];
		context->contactImpulseResultBuffer = buffer;
		context->contactImpulseResultCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsureContactHitEventIdCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	requiredBytes = requiredBytes > 0 ? requiredBytes : sizeof( int );
	if ( context->contactHitEventIdCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->contactHitEventIdCapacity > 0 ? context->contactHitEventIdCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->contactHitEventIdBuffer release];
	context->contactHitEventIdBuffer = buffer;
	context->contactHitEventIdCapacity = capacity;
	return true;
}

const b3MetalContactImpulseResult* b3MetalGetResidentContactImpulseTable(
	const b3MetalContext* context, uint32_t* generation, int* resultCount )
{
	if ( generation != NULL ) *generation = context != NULL ? context->contactImpulseResultGeneration : 0;
	if ( resultCount != NULL ) *resultCount = context != NULL ? context->contactImpulseResultCount : 0;
	return context != NULL && context->contactImpulseResultCount > 0 ? context->contactImpulseResultBuffer.contents : NULL;
}

const int* b3MetalGetResidentHitEventContacts( const b3MetalContext* context, int* contactCount )
{
	int count = context != NULL && context->contactImpulseResultCount > 0 ? context->contactHitEventIdCount : 0;
	if ( contactCount != NULL ) *contactCount = count;
	return count > 0 ? context->contactHitEventIdBuffer.contents : NULL;
}

int b3MetalGetResidentHitEventContactCount( const b3MetalContext* context )
{
	return context != NULL ? context->contactHitEventIdCount : 0;
}

bool b3MetalSyncContactImpulses( const b3MetalContext* context, b3Contact* contact )
{
	if ( context == NULL || contact == NULL || contact->contactId < 0 ||
		contact->contactId >= context->contactImpulseResultCount || contact->manifoldCount != 1 ||
		contact->manifolds == NULL )
	{
		return false;
	}

	const b3MetalContactImpulseResult* result =
		( (const b3MetalContactImpulseResult*)context->contactImpulseResultBuffer.contents ) + contact->contactId;
	b3Manifold* manifold = contact->manifolds;
	if ( result->contactId != (uint32_t)contact->contactId ||
		result->generation != context->contactImpulseResultGeneration ||
		result->contactGeneration != contact->generation || result->pointCount != (uint32_t)manifold->pointCount ||
		result->pointCount < 1 || result->pointCount > 2 )
	{
		return false;
	}

	int resultPointIndices[2] = { B3_NULL_INDEX, B3_NULL_INDEX };
	for ( int pointIndex = 0; pointIndex < manifold->pointCount; ++pointIndex )
	{
		for ( uint32_t resultIndex = 0; resultIndex < result->pointCount; ++resultIndex )
		{
			if ( manifold->points[pointIndex].featureId == result->points[resultIndex].featureId )
			{
				resultPointIndices[pointIndex] = (int)resultIndex;
				break;
			}
		}
		if ( resultPointIndices[pointIndex] == B3_NULL_INDEX ) return false;
	}

	manifold->frictionImpulse = (b3Vec3){ result->frictionX, result->frictionY, result->frictionZ };
	manifold->twistImpulse = result->twistImpulse;
	manifold->rollingImpulse = (b3Vec3){ result->rollingX, result->rollingY, result->rollingZ };
	for ( int pointIndex = 0; pointIndex < manifold->pointCount; ++pointIndex )
	{
		int resultIndex = resultPointIndices[pointIndex];
		b3ManifoldPoint* target = manifold->points + pointIndex;
		target->normalImpulse = result->points[resultIndex].normalImpulse;
		target->totalNormalImpulse = result->points[resultIndex].totalNormalImpulse;
		target->normalVelocity = result->points[resultIndex].normalVelocity;
	}
	return true;
}

void b3MetalInvalidateContactImpulseResults( b3MetalContext* context )
{
	if ( context == NULL ) return;
	context->contactImpulseResultCount = 0;
}

b3ContactConstraintWide* b3MetalGetContactConstraintStorage( b3MetalContext* context, int constraintCount )
{
	if ( context == NULL || constraintCount < 0 )
	{
		return NULL;
	}
	if ( constraintCount == 0 )
	{
		return NULL;
	}
	NSUInteger requiredBytes = (NSUInteger)constraintCount * sizeof( b3ContactConstraintWide );
	if ( b3MetalEnsureContactCapacity( context, requiredBytes ) == false )
	{
		return NULL;
	}
	return context->contactConstraintBuffer.contents;
}

static bool b3MetalEnsureMeshContactCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->meshContactCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->meshContactCapacity > 0 ? context->meshContactCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->meshContactBuffer release];
	context->meshContactBuffer = buffer;
	context->meshContactCapacity = capacity;
	return true;
}

static bool b3MetalEnsureMeshManifoldCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->meshManifoldCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->meshManifoldCapacity > 0 ? context->meshManifoldCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->meshManifoldBuffer release];
	context->meshManifoldBuffer = buffer;
	context->meshManifoldCapacity = capacity;
	return true;
}

b3ContactConstraint* b3MetalGetMeshContactStorage( b3MetalContext* context, int constraintCount )
{
	if ( context == NULL || constraintCount <= 0 ) return NULL;
	NSUInteger bytes = (NSUInteger)constraintCount * sizeof( b3ContactConstraint );
	return b3MetalEnsureMeshContactCapacity( context, bytes ) ? context->meshContactBuffer.contents : NULL;
}

b3ManifoldConstraint* b3MetalGetMeshManifoldStorage( b3MetalContext* context, int manifoldCount )
{
	if ( context == NULL || manifoldCount <= 0 ) return NULL;
	NSUInteger bytes = (NSUInteger)manifoldCount * sizeof( b3ManifoldConstraint );
	return b3MetalEnsureMeshManifoldCapacity( context, bytes ) ? context->meshManifoldBuffer.contents : NULL;
}

static bool b3MetalEnsureDistanceJointCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->distanceJointCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->distanceJointCapacity > 0 ? context->distanceJointCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->distanceJointBuffer release];
	context->distanceJointBuffer = buffer;
	context->distanceJointCapacity = capacity;
	return true;
}

static void b3MetalPackDistanceJoint( b3MetalDistanceJoint* output, const b3JointSim* base )
{
	const b3DistanceJoint* joint = &base->distanceJoint;
	*output = (b3MetalDistanceJoint){
		.indexA = joint->indexA,
		.indexB = joint->indexB,
		.invMassA = base->invMassA,
		.invMassB = base->invMassB,
		.invIA = base->invIA,
		.invIB = base->invIB,
		.constraintSoftness = base->constraintSoftness,
		.anchorA = joint->anchorA,
		.anchorB = joint->anchorB,
		.deltaCenter = joint->deltaCenter,
		.distanceSoftness = joint->distanceSoftness,
		.length = joint->length,
		.hertz = joint->hertz,
		.lowerSpringForce = joint->lowerSpringForce,
		.upperSpringForce = joint->upperSpringForce,
		.minLength = joint->minLength,
		.maxLength = joint->maxLength,
		.maxMotorForce = joint->maxMotorForce,
		.motorSpeed = joint->motorSpeed,
		.impulse = joint->impulse,
		.lowerImpulse = joint->lowerImpulse,
		.upperImpulse = joint->upperImpulse,
		.motorImpulse = joint->motorImpulse,
		.axialMass = joint->axialMass,
		.flags = ( joint->enableSpring ? 1u : 0u ) | ( joint->enableLimit ? 2u : 0u ) |
			( joint->enableMotor ? 4u : 0u ),
	};
}

static void b3MetalUnpackDistanceJoint( b3JointSim* base, const b3MetalDistanceJoint* input )
{
	b3DistanceJoint* joint = &base->distanceJoint;
	joint->impulse = input->impulse;
	joint->lowerImpulse = input->lowerImpulse;
	joint->upperImpulse = input->upperImpulse;
	joint->motorImpulse = input->motorImpulse;
}

static bool b3MetalEnsureParallelJointCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->parallelJointCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->parallelJointCapacity > 0 ? context->parallelJointCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->parallelJointBuffer release];
	context->parallelJointBuffer = buffer;
	context->parallelJointCapacity = capacity;
	return true;
}

static bool b3MetalEnsureJointOverflowCapacity( b3MetalContext* context, NSUInteger requiredBytes )
{
	if ( context->jointOverflowCapacity >= requiredBytes ) return true;
	NSUInteger capacity = context->jointOverflowCapacity > 0 ? context->jointOverflowCapacity : 4096;
	while ( capacity < requiredBytes ) capacity *= 2;
	id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if ( buffer == nil ) return false;
	[context->jointOverflowBuffer release];
	context->jointOverflowBuffer = buffer;
	context->jointOverflowCapacity = capacity;
	return true;
}

static void b3MetalPackParallelJoint( b3MetalParallelJoint* output, const b3JointSim* base )
{
	const b3ParallelJoint* joint = &base->parallelJoint;
	*output = (b3MetalParallelJoint){
		.indexA = joint->indexA,
		.indexB = joint->indexB,
		.invIA = base->invIA,
		.invIB = base->invIB,
		.softness = joint->softness,
		.perpAxisX = joint->perpAxisX,
		.perpAxisY = joint->perpAxisY,
		.quatA = joint->quatA,
		.quatB = joint->quatB,
		.maxTorque = joint->maxTorque,
		.perpImpulse = joint->perpImpulse,
		.fixedRotation = base->fixedRotation ? 1u : 0u,
	};
}

static void b3MetalUnpackParallelJoint( b3JointSim* base, const b3MetalParallelJoint* input )
{
	b3ParallelJoint* joint = &base->parallelJoint;
	joint->perpAxisX = input->perpAxisX;
	joint->perpAxisY = input->perpAxisY;
	joint->perpImpulse = input->perpImpulse;
}

static NSUInteger b3MetalThreadgroupWidth( id<MTLComputePipelineState> pipeline )
{
	NSUInteger width = pipeline.threadExecutionWidth;
	NSUInteger groupWidth = b3MinInt( (int)pipeline.maxTotalThreadsPerThreadgroup, 256 );
	groupWidth -= groupWidth % width;
	return groupWidth > 0 ? groupWidth : width;
}

// Overflow contacts deliberately share bodies, so one GPU thread walks the
// array in deterministic upstream order. This is one dispatch per solver phase,
// rather than one dispatch per contact, and remains in the graph command buffer.
static void b3MetalDispatchOverflowMesh( id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
	id<MTLBuffer> states, id<MTLBuffer> contacts, id<MTLBuffer> manifolds, uint32_t offset, uint32_t count,
	b3MetalContactParams common )
{
	if ( count == 0 )
	{
		return;
	}

	[encoder setComputePipelineState:pipeline];
	[encoder setBuffer:states offset:0 atIndex:0];
	[encoder setBuffer:contacts offset:0 atIndex:1];
	[encoder setBuffer:manifolds offset:0 atIndex:2];
	common.offset = offset;
	common.count = count;
	[encoder setBytes:&common length:sizeof( common ) atIndex:3];
	[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void b3MetalDispatchDistanceJoints( id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
	id<MTLBuffer> states, id<MTLBuffer> joints, uint32_t offset, uint32_t count, b3MetalJointParams common, bool serial )
{
	if ( count == 0 ) return;
	common.offset = offset;
	common.count = count;
	[encoder setComputePipelineState:pipeline];
	[encoder setBuffer:states offset:0 atIndex:0];
	[encoder setBuffer:joints offset:0 atIndex:1];
	[encoder setBytes:&common length:sizeof( common ) atIndex:2];
	NSUInteger threadCount = serial ? 1 : count;
	NSUInteger groupWidth = serial ? 1 : b3MetalThreadgroupWidth( pipeline );
	[encoder dispatchThreads:MTLSizeMake( threadCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( groupWidth, 1, 1 )];
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void b3MetalDispatchJointOverflow( id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
	b3MetalContext* context, uint32_t count, b3MetalJointParams params )
{
	if ( count == 0 ) return;
	params.offset = 0;
	params.count = count;
	[encoder setComputePipelineState:pipeline];
	[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->distanceJointBuffer offset:0 atIndex:1];
	[encoder setBuffer:context->parallelJointBuffer offset:0 atIndex:2];
	[encoder setBuffer:context->jointOverflowBuffer offset:0 atIndex:3];
	[encoder setBytes:&params length:sizeof( params ) atIndex:4];
	[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

static void b3MetalPackBodyProperties( b3MetalBodyProperties* properties, const b3BodySim* sims, int bodyCount )
{
	for ( int i = 0; i < bodyCount; ++i )
	{
		const b3BodySim* sim = sims + i;
		b3MetalBodyProperties* p = properties + i;
		p->qx = sim->transform.q.v.x;
		p->qy = sim->transform.q.v.y;
		p->qz = sim->transform.q.v.z;
		p->qw = sim->transform.q.s;
		p->forceX = sim->force.x;
		p->forceY = sim->force.y;
		p->forceZ = sim->force.z;
		p->torqueX = sim->torque.x;
		p->torqueY = sim->torque.y;
		p->torqueZ = sim->torque.z;
		p->invMass = sim->invMass;
		memcpy( p->invInertiaLocal, &sim->invInertiaLocal, sizeof( p->invInertiaLocal ) );
		memcpy( p->invInertiaWorld, &sim->invInertiaWorld, sizeof( p->invInertiaWorld ) );
		p->linearDamping = sim->linearDamping;
		p->angularDamping = sim->angularDamping;
		p->gravityScale = sim->gravityScale;
	}
}

bool b3MetalStageResidentContactPrepare( b3MetalContext* context, b3Contact* contact )
{
	if ( context == NULL || contact == NULL || context->contactPrepareGeneration == 0 ||
		contact->contactId < 0 || contact->contactId >= context->convexManifoldTableCount ||
		contact->manifoldCount != 1 || contact->manifolds == NULL ||
		contact->manifolds[0].pointCount < 1 || contact->manifolds[0].pointCount > 2 )
	{
		return false;
	}
	NSUInteger tableIndex = (NSUInteger)contact->contactId;
	if ( tableIndex >= context->contactPrepareTableCapacity / sizeof( b3MetalContactPrepareInput ) ) return false;

	b3Manifold* manifold = contact->manifolds;
	const b3MetalContactImpulseResult* previous = NULL;
	if ( context->contactImpulseResultBuffer != nil && contact->contactId < context->contactImpulseResultCount )
	{
		const b3MetalContactImpulseResult* candidate =
			( (const b3MetalContactImpulseResult*)context->contactImpulseResultBuffer.contents ) + contact->contactId;
		if ( candidate->contactId == (uint32_t)contact->contactId &&
			candidate->generation == context->contactImpulseResultGeneration &&
			candidate->contactGeneration == contact->generation && candidate->pointCount >= 1 && candidate->pointCount <= 2 )
		{
			previous = candidate;
			manifold->frictionImpulse = (b3Vec3){ previous->frictionX, previous->frictionY, previous->frictionZ };
			manifold->twistImpulse = previous->twistImpulse;
			manifold->rollingImpulse = (b3Vec3){ previous->rollingX, previous->rollingY, previous->rollingZ };
		}
	}
	b3MetalContactPrepareInput input = {
		.contactId = (uint32_t)contact->contactId,
		.indexA = contact->bodySimIndexA,
		.indexB = contact->bodySimIndexB,
		.generation = context->contactPrepareGeneration,
		.manifold = (uint64_t)(uintptr_t)manifold,
		.friction = contact->friction,
		.restitution = contact->restitution,
		.rollingResistance = contact->rollingResistance,
		.tangentVelocityX = contact->tangentVelocity.x,
		.tangentVelocityY = contact->tangentVelocity.y,
		.tangentVelocityZ = contact->tangentVelocity.z,
		.twistImpulse = manifold->twistImpulse,
		.frictionImpulseX = manifold->frictionImpulse.x,
		.frictionImpulseY = manifold->frictionImpulse.y,
		.frictionImpulseZ = manifold->frictionImpulse.z,
		.rollingImpulseX = manifold->rollingImpulse.x,
		.rollingImpulseY = manifold->rollingImpulse.y,
		.rollingImpulseZ = manifold->rollingImpulse.z,
		.contactGeneration = contact->generation,
	};
	for ( int pointIndex = 0; pointIndex < manifold->pointCount; ++pointIndex )
	{
		const b3ManifoldPoint* point = manifold->points + pointIndex;
		input.points[pointIndex] = (b3MetalContactPreparePoint){
			.anchorAX = point->anchorA.x,
			.anchorAY = point->anchorA.y,
			.anchorAZ = point->anchorA.z,
			.separation = point->separation,
			.anchorBX = point->anchorB.x,
			.anchorBY = point->anchorB.y,
			.anchorBZ = point->anchorB.z,
			.normalImpulse = point->normalImpulse,
			.featureId = point->featureId,
		};
	}
	( (b3MetalContactPrepareInput*)context->contactPrepareTableBuffer.contents )[tableIndex] = input;
	return true;
}

static bool b3MetalPackContactPrepareIndices( b3MetalContext* context, const b3StepContext* stepContext,
	bool* hasRestitution )
{
	if ( hasRestitution != NULL ) *hasRestitution = false;
	if ( context == NULL || stepContext == NULL || stepContext->metalPrepareConvexOnGpu == false ||
		stepContext->wideContactCount <= 0 || context->convexManifoldTableCount <= 0 )
	{
		return false;
	}

	NSUInteger indexCount = (NSUInteger)stepContext->wideContactCount * B3_SIMD_WIDTH;
	if ( indexCount > NSUIntegerMax / sizeof( uint32_t ) ) return false;
	NSUInteger indexBytes = indexCount * sizeof( uint32_t );
	uint64_t graphRevision = stepContext->graph->revision;
	if ( context->contactPrepareIndexBuffer != nil && context->contactPrepareScheduleRevision == graphRevision &&
		context->contactPrepareScheduleWideCount == stepContext->wideContactCount &&
		context->contactPrepareScheduleContactCount == stepContext->metalResidentConvexContactCount )
	{
		stepContext->world->metalLastContactPrepareIndexBytes = indexBytes;
		stepContext->world->metalContactScheduleReuseCount += 1;
		if ( hasRestitution != NULL ) *hasRestitution = stepContext->metalResidentConvexHasRestitution;
		return true;
	}
	if ( b3MetalEnsureContactPrepareIndexCapacity( context, indexBytes ) == false ) return false;
	uint32_t* indices = context->contactPrepareIndexBuffer.contents;
	memset( indices, 0xff, indexBytes );

	int packedCount = 0;
	for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
	{
		const b3WidePrepareSpan* span = stepContext->widePrepareSpans + colorIndex;
		if ( span->count < 0 || span->start < 0 || span->start > stepContext->wideContactCount ) return false;
		if ( span->count > 0 )
		{
			if ( span->contacts == NULL || (NSUInteger)span->count > NSUIntegerMax / sizeof( int ) ) return false;
			NSUInteger destination = (NSUInteger)B3_SIMD_WIDTH * (NSUInteger)span->start;
			if ( destination > indexCount || (NSUInteger)span->count > indexCount - destination ) return false;
			_Static_assert( sizeof( int ) == sizeof( uint32_t ), "Metal contact-id schedule ABI changed" );
			memcpy( indices + destination, span->contacts, (NSUInteger)span->count * sizeof( int ) );
		}
		packedCount += span->count;
	}
	if ( packedCount != stepContext->metalResidentConvexContactCount ) return false;
	context->contactPrepareScheduleRevision = graphRevision;
	context->contactPrepareScheduleWideCount = stepContext->wideContactCount;
	context->contactPrepareScheduleContactCount = stepContext->metalResidentConvexContactCount;
	stepContext->world->metalLastContactPrepareIndexBytes = indexBytes;
	stepContext->world->metalContactSchedulePackCount += 1;
	if ( hasRestitution != NULL ) *hasRestitution = stepContext->metalResidentConvexHasRestitution;
	return true;
}

static void b3MetalPackFinalizeProperties( b3MetalFinalizeProperties* properties, const b3BodySim* sims, int bodyCount,
	const b3World* world )
{
	for ( int i = 0; i < bodyCount; ++i )
	{
		const b3Body* body = world != NULL ? world->bodies.data + sims[i].bodyId : NULL;
		properties[i] = (b3MetalFinalizeProperties){
			.localCenterX = sims[i].localCenter.x,
			.localCenterY = sims[i].localCenter.y,
			.localCenterZ = sims[i].localCenter.z,
			.maxExtentX = sims[i].maxExtent.x,
			.maxExtentY = sims[i].maxExtent.y,
			.maxExtentZ = sims[i].maxExtent.z,
			.centerX = (float)sims[i].center.x,
			.centerY = (float)sims[i].center.y,
			.centerZ = (float)sims[i].center.z,
			.bodyId = sims[i].bodyId,
			.userData = body != NULL ? (uint64_t)(uintptr_t)body->userData : 0,
			.generationWorld = body != NULL ? (uint32_t)world->worldId | ( (uint32_t)body->generation << 16 ) : 0,
		};
#if defined( BOX3D_DOUBLE_PRECISION )
		memcpy( &properties[i].centerXBits, &sims[i].center.x, sizeof( uint64_t ) );
		memcpy( &properties[i].centerYBits, &sims[i].center.y, sizeof( uint64_t ) );
		memcpy( &properties[i].centerZBits, &sims[i].center.z, sizeof( uint64_t ) );
#endif
	}
}

static bool b3MetalEnsureShapeBodyCapacity( b3MetalContext* context, int bodyCount )
{
	if ( context->shapeInputBodyCapacity >= bodyCount ) return true;
	int capacity = context->shapeInputBodyCapacity > 0 ? context->shapeInputBodyCapacity : 64;
	while ( capacity < bodyCount ) capacity *= 2;
	int* bodyIds = realloc( context->shapeInputBodyIds, (size_t)capacity * sizeof( int ) );
	if ( bodyIds == NULL ) return false;
	context->shapeInputBodyIds = bodyIds;
	context->shapeInputBodyCapacity = capacity;
	return true;
}

static void b3MetalAdvanceShapeResultGeneration( b3World* world )
{
	world->metalShapeResultGeneration += 1;
	if ( world->metalShapeResultGeneration == 0 ) world->metalShapeResultGeneration = 1;
}

static int b3MetalPackShapeInputs( b3MetalContext* context, b3StepContext* stepContext )
{
	stepContext->metalShapeResults = NULL;
	stepContext->metalShapeResultCount = 0;
	stepContext->metalEnlargedShapeResults = NULL;
	stepContext->metalEnlargedShapeResultCount = 0;
	stepContext->metalShapeBoundsResident = false;
	stepContext->metalDeferShapeResultApply = false;
	stepContext->metalTreeRefitEligible = false;
	stepContext->metalTreeRefit = false;
	b3World* world = stepContext->world;
	b3BodySim* sims = stepContext->sims;
	int bodyCount = world->solverSets.data[b3_awakeSet].bodySims.count;
	bool treeRefitEligible = world->metalBroadPhaseEnabled && world->userTreeTask == NULL &&
		context->pairTreeBuffer != nil && context->pairTreeRevision == world->broadPhase.treeRevision;
	bool bodyOrderMatches = context->shapeInputCacheValid && context->shapeInputBodyCount == bodyCount;
	for ( int simIndex = 0; simIndex < bodyCount; ++simIndex )
	{
		if ( sims[simIndex].flags & b3_isFast ) treeRefitEligible = false;
		if ( bodyOrderMatches && context->shapeInputBodyIds[simIndex] != sims[simIndex].bodyId )
		{
			bodyOrderMatches = false;
		}
	}

	bool boundsResident = context->shapeBoundsRevision == world->broadPhase.treeRevision &&
		context->shapeBoundsCount == context->shapeInputCount;
	if ( bodyOrderMatches && boundsResident && context->shapeInputCount > 0 )
	{
		stepContext->metalShapeBoundsResident = true;
		stepContext->metalDeferShapeResultApply = world->enableContinuous == false && world->contacts.count == 0 &&
			world->sensors.count == 0 && context->shapeInputAllMasksDisabled;
		stepContext->metalTreeRefitEligible = treeRefitEligible;
		world->metalShapeInputReuseCount += 1;
		return context->shapeInputCount;
	}
	if ( world->metalShapeCpuBoundsStale && b3MetalSyncAllShapeBounds( context, world ) == false )
	{
		context->shapeInputCacheValid = false;
		world->metalShapeFallbackCount += 1;
		return 0;
	}

	int shapeCount = 0;
	for ( int i = 0; i < bodyCount; ++i )
	{
		shapeCount += world->bodies.data[sims[i].bodyId].shapeCount;
	}
	if ( shapeCount == 0 )
	{
		context->shapeInputCacheValid = false;
		return 0;
	}
	if ( b3MetalEnsureShapeBodyCapacity( context, bodyCount ) == false ||
		b3MetalEnsureShapeCapacity( context, (NSUInteger)shapeCount * sizeof( b3MetalShapeInput ),
		(NSUInteger)shapeCount * sizeof( b3MetalShapeAABBResult ) ) == false )
	{
		context->shapeInputCacheValid = false;
		world->metalShapeFallbackCount += 1;
		return 0;
	}

	b3MetalShapeInput* inputs = context->shapeInputBuffer.contents;
	bool allMasksDisabled = true;
	int outputIndex = 0;
	for ( int simIndex = 0; simIndex < bodyCount; ++simIndex )
	{
		context->shapeInputBodyIds[simIndex] = sims[simIndex].bodyId;
		b3Body* body = world->bodies.data + sims[simIndex].bodyId;
		int shapeId = body->headShapeId;
		while ( shapeId != B3_NULL_INDEX )
		{
			b3Shape* shape = world->shapes.data + shapeId;
			if ( shape->filter.maskBits != 0 ) allMasksDisabled = false;
			shape->metalResultIndex = outputIndex;
			b3MetalShapeInput* input = inputs + outputIndex++;
			*input = (b3MetalShapeInput){
				.bodyIndex = (uint32_t)simIndex,
				.shapeId = (uint32_t)shapeId,
				.type = (uint32_t)shape->type,
				.proxyKey = shape->proxyKey,
				.margin = shape->aabbMargin,
				.fatLowerX = (float)shape->fatAABB.lowerBound.x,
				.fatLowerY = (float)shape->fatAABB.lowerBound.y,
				.fatLowerZ = (float)shape->fatAABB.lowerBound.z,
				.fatUpperX = (float)shape->fatAABB.upperBound.x,
				.fatUpperY = (float)shape->fatAABB.upperBound.y,
				.fatUpperZ = (float)shape->fatAABB.upperBound.z,
			};
			if ( shape->type == b3_sphereShape )
			{
				input->point1X = shape->sphere.center.x;
				input->point1Y = shape->sphere.center.y;
				input->point1Z = shape->sphere.center.z;
				input->point2X = shape->sphere.center.x;
				input->point2Y = shape->sphere.center.y;
				input->point2Z = shape->sphere.center.z;
				input->radius = shape->sphere.radius;
			}
			else if ( shape->type == b3_capsuleShape )
			{
				input->point1X = shape->capsule.center1.x;
				input->point1Y = shape->capsule.center1.y;
				input->point1Z = shape->capsule.center1.z;
				input->point2X = shape->capsule.center2.x;
				input->point2Y = shape->capsule.center2.y;
				input->point2Z = shape->capsule.center2.z;
				input->radius = shape->capsule.radius;
			}
			else
			{
				b3AABB localAABB = b3ComputeShapeAABB( shape, b3Transform_identity );
				input->point1X = (float)localAABB.lowerBound.x;
				input->point1Y = (float)localAABB.lowerBound.y;
				input->point1Z = (float)localAABB.lowerBound.z;
				input->point2X = (float)localAABB.upperBound.x;
				input->point2Y = (float)localAABB.upperBound.y;
				input->point2Z = (float)localAABB.upperBound.z;
			}
			shapeId = shape->nextShapeId;
		}
	}
	B3_ASSERT( outputIndex == shapeCount );
	context->shapeInputBodyCount = bodyCount;
	context->shapeInputCount = shapeCount;
	context->shapeInputAllMasksDisabled = allMasksDisabled;
	context->shapeInputCacheValid = true;
	stepContext->metalDeferShapeResultApply = world->enableContinuous == false && world->contacts.count == 0 &&
		world->sensors.count == 0 && allMasksDisabled;
	stepContext->metalTreeRefitEligible = treeRefitEligible;
	world->metalShapeInputPackCount += 1;
	return shapeCount;
}

static void b3MetalEncodeShapeFinalization( b3MetalContext* context, id<MTLComputeCommandEncoder> encoder,
	b3StepContext* stepContext, int shapeCount )
{
	if ( shapeCount == 0 ) return;
	struct { uint32_t shapeCount; float extra; uint32_t padding[2]; } params =
		{ (uint32_t)shapeCount, B3_SPECULATIVE_DISTANCE,
		  { stepContext != NULL && stepContext->metalShapeBoundsResident ? 1u : 0u, 0 } };
	id<MTLComputePipelineState> pipeline = context->finalizeShapesPipeline;
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:pipeline];
	[encoder setBuffer:context->shapeInputBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:1];
	[encoder setBuffer:context->shapeResultBuffer offset:0 atIndex:2];
	[encoder setBytes:&params length:sizeof( params ) atIndex:3];
	[encoder setBuffer:context->finalizePropertiesBuffer offset:0 atIndex:4];
	[encoder dispatchThreads:MTLSizeMake( (NSUInteger)shapeCount, 1, 1 )
		threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pipeline ), 1, 1 )];
}

static bool b3MetalEncodeFinalizeReadback( b3MetalContext* context, id<MTLCommandBuffer> commandBuffer, int bodyCount )
{
	if ( bodyCount <= 0 ) return true;
	id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
	if ( blit == nil ) return false;
	[blit copyFromBuffer:context->finalizeResultBuffer sourceOffset:0 toBuffer:context->finalizeReadbackBuffer
		destinationOffset:0 size:(NSUInteger)bodyCount * sizeof( b3MetalFinalizeResult )];
	[blit endEncoding];
	return true;
}

static bool b3MetalEncodeFullShapeReadback( b3MetalContext* context, id<MTLCommandBuffer> commandBuffer, int shapeCount )
{
	if ( shapeCount <= 0 ) return true;
	id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
	if ( blit == nil ) return false;
	[blit copyFromBuffer:context->shapeResultBuffer sourceOffset:0 toBuffer:context->shapeReadbackBuffer
		destinationOffset:0 size:(NSUInteger)shapeCount * sizeof( b3MetalShapeAABBResult )];
	[blit endEncoding];
	return true;
}

static bool b3MetalReadbackShapeRange( b3MetalContext* context, NSUInteger offset, NSUInteger size )
{
	if ( context == NULL || size == 0 ) return false;
	id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
	id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
	if ( commandBuffer == nil || blit == nil ) return false;
	[blit copyFromBuffer:context->shapeResultBuffer sourceOffset:offset toBuffer:context->shapeReadbackBuffer
		destinationOffset:offset size:size];
	[blit endEncoding];
	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];
	return commandBuffer.status == MTLCommandBufferStatusCompleted;
}

static bool b3MetalEncodePairTreeRefit( b3MetalContext* context, id<MTLComputeCommandEncoder> encoder,
	b3StepContext* stepContext, int shapeCount )
{
	if ( shapeCount == 0 || stepContext == NULL || stepContext->metalTreeRefitEligible == false ) return false;
	NSUInteger scanWidth = context->shapeScanBlocksPipeline.threadExecutionWidth;
	if ( context->shapeScanBlocksPipeline.maxTotalThreadsPerThreadgroup < 256 || scanWidth < 8 || 256 % scanWidth != 0 )
	{
		return false;
	}
	uint32_t blockCount = ( (uint32_t)shapeCount + 255u ) / 256u;
	struct
	{
		uint32_t shapeCount, blockCount, padding0, padding1;
	} compactParams = { (uint32_t)shapeCount, blockCount, 0, 0 };
	_Static_assert( sizeof( compactParams ) == 16, "Metal shape-compact parameter ABI changed" );

	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:context->shapeScanBlocksPipeline];
	[encoder setBuffer:context->shapeResultBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->shapeBlockBuffer offset:0 atIndex:1];
	[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:2];
	[encoder dispatchThreadgroups:MTLSizeMake( blockCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 256, 1, 1 )];

	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:context->shapePrefixPipeline];
	[encoder setBuffer:context->shapeBlockBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->shapeSummaryBuffer offset:0 atIndex:1];
	[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:2];
	[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];

	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:context->shapeScatterPipeline];
	[encoder setBuffer:context->shapeInputBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->shapeResultBuffer offset:0 atIndex:1];
	[encoder setBuffer:context->shapeBlockBuffer offset:0 atIndex:2];
	[encoder setBuffer:context->shapeCompactBuffer offset:0 atIndex:3];
	[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:4];
	[encoder dispatchThreads:MTLSizeMake( (NSUInteger)shapeCount, 1, 1 )
		threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->shapeScatterPipeline ), 1, 1 )];

	struct
	{
		uint32_t offset0, offset1, offset2, padding;
	} offsets = { context->pairTreeOffsets[0], context->pairTreeOffsets[1], context->pairTreeOffsets[2], 0 };
	_Static_assert( sizeof( offsets ) == 16, "Metal tree-offset ABI changed" );
	[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
	[encoder setComputePipelineState:context->pairUpdateLeavesPipeline];
	[encoder setBuffer:context->shapeInputBuffer offset:0 atIndex:0];
	[encoder setBuffer:context->shapeResultBuffer offset:0 atIndex:1];
	[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:2];
	[encoder setBytes:&offsets length:sizeof( offsets ) atIndex:3];
	[encoder dispatchThreads:MTLSizeMake( (NSUInteger)shapeCount, 1, 1 )
		threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairUpdateLeavesPipeline ), 1, 1 )];

	for ( int treeIndex = b3_kinematicBody; treeIndex <= b3_dynamicBody; ++treeIndex )
	{
		uint32_t nodeCount = context->pairTreeNodeCounts[treeIndex];
		for ( uint32_t height = 1; height <= context->pairTreeHeights[treeIndex]; ++height )
		{
			struct
			{
				uint32_t nodeOffset, nodeCount, targetHeight, padding;
			} params = { context->pairTreeOffsets[treeIndex], nodeCount, height, 0 };
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			[encoder setComputePipelineState:context->pairRefitPipeline];
			[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:0];
			[encoder setBytes:&params length:sizeof( params ) atIndex:1];
			[encoder dispatchThreads:MTLSizeMake( nodeCount, 1, 1 ) threadsPerThreadgroup:
				MTLSizeMake( b3MetalThreadgroupWidth( context->pairRefitPipeline ), 1, 1 )];
		}
	}
	return true;
}

static bool b3MetalHasRestitution( const b3ContactConstraintWide* constraints, int count )
{
	for ( int i = 0; i < count; ++i )
	{
		float lanes[B3_SIMD_WIDTH];
		b3StoreW( lanes, constraints[i].restitution );
		if ( lanes[0] != 0.0f || lanes[1] != 0.0f || lanes[2] != 0.0f || lanes[3] != 0.0f )
		{
			return true;
		}
	}
	return false;
}

static bool b3MetalHasMeshRestitution( const b3ContactConstraint* constraints, int count )
{
	for ( int i = 0; i < count; ++i )
	{
		if ( constraints[i].restitution != 0.0f ) return true;
	}
	return false;
}

bool b3MetalIntegratePositions( b3MetalContext* context, b3BodyState* states, int bodyCount, float h,
								float maxLinearSpeed, float maxAngularSpeed, b3MetalDispatchStats* stats )
{
	if ( context != NULL ) context->bodyPropertiesResidentCount = 0;
	if ( stats != NULL )
	{
		*stats = (b3MetalDispatchStats){ .bodyCount = bodyCount };
	}

	if ( context == NULL || states == NULL || bodyCount < 0 )
	{
		return false;
	}
	if ( bodyCount == 0 )
	{
		return true;
	}

	@autoreleasepool
	{
		NSUInteger byteCount = (NSUInteger)bodyCount * sizeof( b3BodyState );
		if ( b3MetalEnsureBodyCapacity( context, byteCount ) == false )
		{
			return false;
		}
		context->bodyStateResidentCount = 0;
		memcpy( context->bodyStateBuffer.contents, states, byteCount );

		b3MetalIntegrateParams params = {
			.bodyCount = (uint32_t)bodyCount,
			.h = h,
			.maxLinearSpeed = maxLinearSpeed,
			.maxAngularSpeed = maxAngularSpeed,
		};

		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		[encoder setComputePipelineState:context->integratePositionsPipeline];
		[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
		[encoder setBytes:&params length:sizeof( params ) atIndex:1];

		NSUInteger width = context->integratePositionsPipeline.threadExecutionWidth;
		NSUInteger maxWidth = context->integratePositionsPipeline.maxTotalThreadsPerThreadgroup;
		NSUInteger groupWidth = maxWidth < 256 ? maxWidth : 256;
		groupWidth -= groupWidth % width;
		if ( groupWidth == 0 )
		{
			groupWidth = width;
		}
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( groupWidth, 1, 1 )];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}

		memcpy( states, context->bodyStateBuffer.contents, byteCount );
		if ( stats != NULL && commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
		{
			stats->gpuMilliseconds = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
		}
		return true;
	}
}

static bool b3MetalPrepareBodyTransformDeviceRefresh( b3MetalContext* context, const b3World* world );
static void b3MetalCommitBodyTransformDeviceRefresh( b3MetalContext* context, b3World* world, int bodyCount );

bool b3MetalIntegrateUnconstrainedSubsteps( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims,
	int bodyCount, int subStepCount, float h, b3Vec3 gravity, float maxLinearSpeed, float maxAngularSpeed,
	float invTimeStep, const b3MetalFinalizeResult** finalizeResults, b3StepContext* finalizationContext,
	b3MetalDispatchStats* stats )
{
	if ( finalizeResults != NULL )
	{
		*finalizeResults = NULL;
	}
	if ( stats != NULL )
	{
		*stats = (b3MetalDispatchStats){ .bodyCount = bodyCount };
	}
	if ( context == NULL || states == NULL || sims == NULL || bodyCount < 0 || subStepCount < 1 )
	{
		if ( context != NULL ) context->bodyPropertiesResidentCount = 0;
		return false;
	}
		if ( bodyCount == 0 )
	{
		return true;
	}

	@autoreleasepool
	{
		NSUInteger stateByteCount = (NSUInteger)bodyCount * sizeof( b3BodyState );
		NSUInteger propertiesByteCount = (NSUInteger)bodyCount * sizeof( b3MetalBodyProperties );
		NSUInteger finalizationByteCount = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeResult );
		NSUInteger finalizePropertyByteCount = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeProperties );
		NSUInteger bodyMoveByteCount = (NSUInteger)bodyCount * sizeof( b3MetalBodyMoveResult );
		if ( b3MetalEnsureBodyCapacity( context, stateByteCount ) == false ||
			 b3MetalEnsurePropertiesCapacity( context, propertiesByteCount ) == false ||
			 ( finalizeResults != NULL && ( b3MetalEnsureFinalizeResultCapacity( context, finalizationByteCount ) == false ||
			   b3MetalEnsureFinalizePropertiesCapacity( context, finalizePropertyByteCount ) == false ) ) )
		{
			context->bodyPropertiesResidentCount = 0;
			return false;
		}

		bool omitFinalizeReadback = finalizeResults != NULL && finalizationContext != NULL &&
			finalizationContext->world->enableSleep == false && finalizationContext->world->enableContinuous == false;
		bool publishBodyTransforms = omitFinalizeReadback &&
			b3MetalPrepareBodyTransformDeviceRefresh( context, finalizationContext->world );
		context->bodyMoveResultCount = 0;
		if ( publishBodyTransforms && b3MetalEnsureBodyMoveCapacity( context, bodyMoveByteCount ) == false )
		{
			context->bodyPropertiesResidentCount = 0;
			return false;
		}
		bool reuseBodyStates = publishBodyTransforms && context->bodyStateResidentCount == bodyCount &&
			context->bodyStateResidentRevision == finalizationContext->world->metalBodyStateRevision;
		if ( finalizationContext != NULL )
		{
			finalizationContext->world->metalBodyStateRevisionCheckCount += publishBodyTransforms ? 1 : 0;
		}
		bool reuseBodyProperties = publishBodyTransforms && context->bodyPropertiesResidentCount == bodyCount &&
			context->bodyPropertiesResidentRevision == finalizationContext->world->metalBodyPropertyRevision;
		// A failed command must not leave the previous generation reusable.
		context->bodyStateResidentCount = 0;
		context->bodyPropertiesResidentCount = 0;
		if ( reuseBodyStates == false )
		{
			memcpy( context->bodyStateBuffer.contents, states, stateByteCount );
		}
		if ( finalizationContext != NULL )
		{
			finalizationContext->world->metalBodyStateReuseCount += reuseBodyStates ? 1 : 0;
			finalizationContext->world->metalBodyStateUploadCount += reuseBodyStates ? 0 : 1;
			finalizationContext->world->metalLastBodyStateUploadBytes = reuseBodyStates ? 0 : stateByteCount;
		}
		if ( reuseBodyProperties == false )
		{
			b3MetalPackBodyProperties( context->bodyPropertiesBuffer.contents, sims, bodyCount );
		}
		if ( finalizationContext != NULL )
		{
			finalizationContext->world->metalBodyPropertyReuseCount += reuseBodyProperties ? 1 : 0;
			finalizationContext->world->metalBodyPropertyUploadCount += reuseBodyProperties ? 0 : 1;
			finalizationContext->world->metalLastBodyPropertyUploadBytes =
				reuseBodyProperties ? 0 : propertiesByteCount;
		}
		if ( finalizeResults != NULL )
		{
			b3MetalPackFinalizeProperties( context->finalizePropertiesBuffer.contents, sims, bodyCount,
				finalizationContext != NULL ? finalizationContext->world : NULL );
		}
		int shapeCount = finalizationContext != NULL ? b3MetalPackShapeInputs( context, finalizationContext ) : 0;
		bool treeRefitEncoded = false;
		bool shapeReadbackEncoded = false;

		b3MetalFusedParams params = {
			.bodyCount = (uint32_t)bodyCount,
			.h = h,
			.maxLinearSpeed = maxLinearSpeed,
			.maxAngularSpeed = maxAngularSpeed,
			.gravityX = gravity.x,
			.gravityY = gravity.y,
			.gravityZ = gravity.z,
			.integratePosition = 1,
		};

		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		id<MTLComputePipelineState> pipeline = context->integrateUnconstrainedPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:1];
		[encoder setBytes:&params length:sizeof( params ) atIndex:2];

		NSUInteger width = pipeline.threadExecutionWidth;
		NSUInteger maxWidth = pipeline.maxTotalThreadsPerThreadgroup;
		NSUInteger groupWidth = maxWidth < 256 ? maxWidth : 256;
		groupWidth -= groupWidth % width;
		groupWidth = groupWidth > 0 ? groupWidth : width;
		for ( int subStepIndex = 0; subStepIndex < subStepCount; ++subStepIndex )
		{
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( groupWidth, 1, 1 )];
			if ( subStepIndex + 1 < subStepCount )
			{
				[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			}
		}
		if ( finalizeResults != NULL )
		{
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			struct { uint32_t bodyCount; float invTimeStep; uint32_t transformCount, publishTransforms; } finalizeParams = {
				(uint32_t)bodyCount, invTimeStep,
				publishBodyTransforms ? (uint32_t)finalizationContext->world->bodies.count : 0u,
				publishBodyTransforms ? 1u : 0u,
			};
			id<MTLComputePipelineState> finalizePipeline = context->finalizeBodiesPipeline;
			[encoder setComputePipelineState:finalizePipeline];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->finalizePropertiesBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:3];
			[encoder setBytes:&finalizeParams length:sizeof( finalizeParams ) atIndex:4];
			[encoder setBuffer:( publishBodyTransforms ? context->convexBodyTransformBuffer : context->finalizeResultBuffer )
				offset:0 atIndex:5];
			[encoder setBuffer:( publishBodyTransforms ? context->bodyMoveResultBuffer : context->finalizeResultBuffer )
				offset:0 atIndex:6];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( finalizePipeline ), 1, 1 )];
			b3MetalEncodeShapeFinalization( context, encoder, finalizationContext, shapeCount );
			treeRefitEncoded = b3MetalEncodePairTreeRefit( context, encoder, finalizationContext, shapeCount );
		}
		[encoder endEncoding];
		if ( finalizeResults != NULL && omitFinalizeReadback == false &&
			b3MetalEncodeFinalizeReadback( context, commandBuffer, bodyCount ) == false )
		{
			return false;
		}
		if ( shapeCount > 0 && ( treeRefitEncoded == false || finalizationContext->metalDeferShapeResultApply == false ) )
		{
			if ( b3MetalEncodeFullShapeReadback( context, commandBuffer, shapeCount ) == false ) return false;
			shapeReadbackEncoded = true;
		}
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}
		if ( publishBodyTransforms )
		{
			b3MetalCommitBodyTransformDeviceRefresh( context, finalizationContext->world, bodyCount );
			context->bodyStateResidentRevision = finalizationContext->world->metalBodyStateRevision;
			context->bodyPropertiesResidentCount = bodyCount;
			context->bodyPropertiesResidentRevision = finalizationContext->world->metalBodyPropertyRevision;
			context->bodyMoveResultCount = bodyCount;
			context->bodyMoveResultStepIndex = finalizationContext->world->stepIndex;
			finalizationContext->metalBodyStatesFinalizedOnDevice = true;
			finalizationContext->metalBodyMoveEventsOnDevice = true;
		}
		else
		{
			context->bodyStateResidentCount = 0;
			context->bodyPropertiesResidentCount = 0;
		}

		memcpy( states, context->bodyStateBuffer.contents, stateByteCount );
		if ( finalizationContext != NULL )
		{
			finalizationContext->world->metalLastBodyStateReadbackBytes = stateByteCount;
		}
		if ( finalizeResults != NULL )
		{
			if ( omitFinalizeReadback )
			{
				finalizationContext->metalFinalizationDeviceOnly = true;
			}
			else
			{
				*finalizeResults = context->finalizeReadbackBuffer.contents;
			}
		}
		if ( finalizationContext != NULL && shapeCount > 0 )
		{
			b3MetalAdvanceShapeResultGeneration( finalizationContext->world );
			context->shapeBoundsCount = shapeCount;
			finalizationContext->world->metalShapeBoundsResidentDispatchCount +=
				finalizationContext->metalShapeBoundsResident ? 1 : 0;
			finalizationContext->metalShapeResults = shapeReadbackEncoded ? context->shapeReadbackBuffer.contents : NULL;
			finalizationContext->metalShapeResultCount = shapeCount;
			finalizationContext->world->metalLastShapeResultCount = shapeCount;
			finalizationContext->world->metalLastEnlargedShapeResultCount = 0;
			if ( treeRefitEncoded )
			{
				b3MetalPairSummary* summary = context->shapeSummaryBuffer.contents;
				if ( summary->totalCount <= (uint64_t)shapeCount )
				{
					finalizationContext->metalEnlargedShapeResults = context->shapeCompactBuffer.contents;
					finalizationContext->metalEnlargedShapeResultCount = (int)summary->totalCount;
					finalizationContext->world->metalLastEnlargedShapeResultCount = (int)summary->totalCount;
					finalizationContext->world->metalShapeCompactDispatchCount += 1;
				}
				else
				{
					treeRefitEncoded = false;
					if ( shapeReadbackEncoded == false )
					{
						if ( b3MetalReadbackShapeRange( context, 0,
							(NSUInteger)shapeCount * sizeof( b3MetalShapeAABBResult ) ) == false ) return false;
						shapeReadbackEncoded = true;
						finalizationContext->metalShapeResults = context->shapeReadbackBuffer.contents;
					}
				}
			}
			finalizationContext->metalTreeRefit = treeRefitEncoded;
			finalizationContext->world->metalShapeDispatchCount += 1;
			finalizationContext->world->metalPairTreeRefitCount += treeRefitEncoded ? 1 : 0;
		}
		if ( stats != NULL && commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
		{
			stats->gpuMilliseconds = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
		}
		return true;
	}
}

bool b3MetalIntegrateUnconstrained( b3MetalContext* context, b3BodyState* states, const b3BodySim* sims, int bodyCount,
									float h, b3Vec3 gravity, float maxLinearSpeed, float maxAngularSpeed,
									b3MetalDispatchStats* stats )
{
	return b3MetalIntegrateUnconstrainedSubsteps( context, states, sims, bodyCount, 1, h, gravity, maxLinearSpeed,
		maxAngularSpeed, 0.0f, NULL, NULL, stats );
}

bool b3MetalFinalizeBodies( b3MetalContext* context, const b3BodyState* states, const b3BodySim* sims,
	int bodyCount, float invTimeStep, bool statesAreResident, const b3MetalFinalizeResult** results,
	b3MetalDispatchStats* stats )
{
	if ( context != NULL ) context->bodyPropertiesResidentCount = 0;
	if ( results != NULL )
	{
		*results = NULL;
	}
	if ( stats != NULL )
	{
		*stats = (b3MetalDispatchStats){ .bodyCount = bodyCount };
	}
	if ( context == NULL || states == NULL || sims == NULL || results == NULL || bodyCount < 0 )
	{
		return false;
	}
	if ( bodyCount == 0 )
	{
		return true;
	}

	@autoreleasepool
	{
		NSUInteger stateBytes = (NSUInteger)bodyCount * sizeof( b3BodyState );
		NSUInteger propertyBytes = (NSUInteger)bodyCount * sizeof( b3MetalBodyProperties );
		NSUInteger resultBytes = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeResult );
		NSUInteger finalizePropertyBytes = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeProperties );
		if ( b3MetalEnsureBodyCapacity( context, stateBytes ) == false ||
			 b3MetalEnsurePropertiesCapacity( context, propertyBytes ) == false ||
			 b3MetalEnsureFinalizeResultCapacity( context, resultBytes ) == false ||
			 b3MetalEnsureFinalizePropertiesCapacity( context, finalizePropertyBytes ) == false )
		{
			return false;
		}
		context->bodyStateResidentCount = 0;
		if ( statesAreResident == false )
		{
			memcpy( context->bodyStateBuffer.contents, states, stateBytes );
		}
		b3MetalPackBodyProperties( context->bodyPropertiesBuffer.contents, sims, bodyCount );
		b3MetalPackFinalizeProperties( context->finalizePropertiesBuffer.contents, sims, bodyCount, NULL );

		struct
		{
			uint32_t bodyCount;
			float invTimeStep;
			uint32_t padding[2];
		} params = { (uint32_t)bodyCount, invTimeStep, { 0, 0 } };

		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		id<MTLComputePipelineState> pipeline = context->finalizeBodiesPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->finalizePropertiesBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:3];
		[encoder setBytes:&params length:sizeof( params ) atIndex:4];
		[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:6];
		NSUInteger groupWidth = b3MetalThreadgroupWidth( pipeline );
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( groupWidth, 1, 1 )];
		[encoder endEncoding];
		if ( b3MetalEncodeFinalizeReadback( context, commandBuffer, bodyCount ) == false )
		{
			return false;
		}
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}

		*results = context->finalizeReadbackBuffer.contents;
		if ( stats != NULL && commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
		{
			stats->gpuMilliseconds = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
		}
		return true;
	}
}

bool b3MetalGeneratePairCandidates( b3MetalContext* context, const b3World* world, const int* moveArray, int moveCount,
	const b3MetalPairQueryRecord** recordsOut,
	const b3MetalPairCandidate** candidatesOut, int* candidateCountOut, b3MetalDispatchStats* stats )
{
	if ( recordsOut != NULL ) *recordsOut = NULL;
	if ( candidatesOut != NULL ) *candidatesOut = NULL;
	if ( candidateCountOut != NULL ) *candidateCountOut = 0;
	if ( stats != NULL ) *stats = (b3MetalDispatchStats){ .bodyCount = moveCount };
	if ( context == NULL || world == NULL || moveArray == NULL || moveCount < 0 || recordsOut == NULL ||
		 candidatesOut == NULL || candidateCountOut == NULL )
	{
		return false;
	}
	if ( moveCount == 0 ) return true;
	const b3BroadPhase* broadPhase = &world->broadPhase;
	NSUInteger scanExecutionWidth = context->pairScanBlocksPipeline.threadExecutionWidth;
	if ( context->pairScanBlocksPipeline.maxTotalThreadsPerThreadgroup < 256 || scanExecutionWidth < 8 ||
		 256 % scanExecutionWidth != 0 )
	{
		return false;
	}

	// The shader uses a 64-entry private stack. Refuse the route before dispatch
	// whenever the current tree topology cannot be represented exactly.
	for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
	{
		if ( b3DynamicTree_GetHeight( broadPhase->trees + treeIndex ) >= 63 ) return false;
	}

	@autoreleasepool
	{
		NSUInteger moveBytes = (NSUInteger)moveCount * sizeof( int );
		NSUInteger recordBytes = (NSUInteger)moveCount * sizeof( b3MetalPairQueryRecord );
		NSUInteger pairBlockCount = ( (NSUInteger)moveCount + 255 ) / 256;
		NSUInteger blockBytes = pairBlockCount * sizeof( b3MetalPairBlock );
		uint32_t nodeOffsets[b3_bodyTypeCount] = { 0 };
		uint64_t totalNodeCount = 0;
		for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
		{
			nodeOffsets[treeIndex] = (uint32_t)totalNodeCount;
			totalNodeCount += (uint64_t)broadPhase->trees[treeIndex].nodeCapacity;
		}
		if ( totalNodeCount > UINT32_MAX ) return false;
		NSUInteger treeBytes = (NSUInteger)totalNodeCount * sizeof( b3TreeNode );
		NSUInteger movedBytes = (NSUInteger)totalNodeCount * sizeof( uint32_t );
		int shapeCount = world->shapes.count;
		if ( shapeCount <= 0 || (NSUInteger)shapeCount > NSUIntegerMax / sizeof( b3MetalPairShape ) ) return false;
		NSUInteger shapeBytes = (NSUInteger)shapeCount * sizeof( b3MetalPairShape );
		uint32_t pairSetCapacity = broadPhase->pairSet.capacity;
		if ( pairSetCapacity == 0 || ( pairSetCapacity & ( pairSetCapacity - 1 ) ) != 0 ||
			 (NSUInteger)pairSetCapacity > NSUIntegerMax / sizeof( b3SetItem ) ) return false;
		NSUInteger setBytes = (NSUInteger)pairSetCapacity * sizeof( b3SetItem );
		uint64_t candidateLimit64 = 64ull * (uint64_t)moveCount;
		uint32_t candidateLimit = candidateLimit64 < INT32_MAX ? (uint32_t)candidateLimit64 : INT32_MAX;
		uint64_t initialCandidateCount = 4ull * (uint64_t)moveCount;
		if ( initialCandidateCount > candidateLimit ) initialCandidateCount = candidateLimit;
		if ( initialCandidateCount > NSUIntegerMax / sizeof( b3MetalPairCandidate ) ) return false;
		NSUInteger initialCandidateBytes = (NSUInteger)initialCandidateCount * sizeof( b3MetalPairCandidate );
		if ( b3MetalEnsurePairCapacity( context, moveBytes, treeBytes, movedBytes, shapeBytes, setBytes, recordBytes,
			 initialCandidateBytes, blockBytes ) == false )
		{
			return false;
		}
		for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
		{
			const b3DynamicTree* tree = broadPhase->trees + treeIndex;
			context->pairTreeOffsets[treeIndex] = nodeOffsets[treeIndex];
			context->pairTreeNodeCounts[treeIndex] = (uint32_t)tree->nodeCapacity;
			context->pairTreeHeights[treeIndex] = tree->root == B3_NULL_INDEX ? 0 : tree->nodes[tree->root].height;
		}

		memcpy( context->pairMoveBuffer.contents, moveArray, moveBytes );
		if ( context->pairShapeRevision != world->metalPairShapeRevision )
		{
			b3MetalPairShape* pairShapes = context->pairShapeBuffer.contents;
			for ( int shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex )
			{
				const b3Shape* shape = world->shapes.data + shapeIndex;
				if ( shape->id == B3_NULL_INDEX )
				{
					pairShapes[shapeIndex] = (b3MetalPairShape){ .bodyId = B3_NULL_INDEX, .sensorIndex = B3_NULL_INDEX };
					continue;
				}
				pairShapes[shapeIndex] = (b3MetalPairShape){
					.bodyId = shape->bodyId,
					.sensorIndex = shape->sensorIndex,
					.groupIndex = shape->filter.groupIndex,
					.type = (uint32_t)shape->type,
					.categoryBits = shape->filter.categoryBits,
					.maskBits = shape->filter.maskBits,
				};
			}
			context->pairShapeRevision = world->metalPairShapeRevision;
			if ( stats != NULL ) stats->metadataUploadCount = 1;
		}
		if ( context->pairSetRevision != broadPhase->pairSetRevision )
		{
			memcpy( context->pairSetBuffer.contents, broadPhase->pairSet.items, setBytes );
			context->pairSetRevision = broadPhase->pairSetRevision;
			if ( stats != NULL ) stats->pairSetUploadCount = 1;
		}
		uint32_t* movedNodes = context->pairMovedBuffer.contents;
		memset( movedNodes, 0, movedBytes );
		for ( int moveIndex = 0; moveIndex < moveCount; ++moveIndex )
		{
			int proxyKey = moveArray[moveIndex];
			b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
			int proxyId = B3_PROXY_ID( proxyKey );
			if ( proxyType < 0 || proxyType >= b3_bodyTypeCount || proxyId < 0 ||
				 proxyId >= broadPhase->trees[proxyType].nodeCapacity )
			{
				return false;
			}
			movedNodes[nodeOffsets[proxyType] + (uint32_t)proxyId] = 1;
		}
		if ( context->pairTreeRevision != broadPhase->treeRevision )
		{
			b3TreeNode* nodeDestination = context->pairTreeBuffer.contents;
			for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
			{
				const b3DynamicTree* tree = broadPhase->trees + treeIndex;
				memcpy( nodeDestination + nodeOffsets[treeIndex], tree->nodes,
					(NSUInteger)tree->nodeCapacity * sizeof( b3TreeNode ) );
			}
			context->pairTreeRevision = broadPhase->treeRevision;
			if ( stats != NULL ) stats->treeUploadCount = 1;
		}
		memset( context->pairRecordBuffer.contents, 0, recordBytes );
		memset( context->pairSummaryBuffer.contents, 0, sizeof( b3MetalPairSummary ) );

		struct
		{
			int root0, root1, root2;
			uint32_t offset0, offset1, offset2, moveCount, writeCandidates, shapeCount, pairCapacity, padding1, padding2;
		} params = {
			broadPhase->trees[0].root, broadPhase->trees[1].root, broadPhase->trees[2].root,
			nodeOffsets[0], nodeOffsets[1], nodeOffsets[2], (uint32_t)moveCount, 0, (uint32_t)shapeCount, pairSetCapacity, 0, 0,
		};
		_Static_assert( sizeof( params ) == 48, "Metal pair parameter ABI changed" );
		uint64_t availableCandidateCount64 = context->pairCandidateCapacity / sizeof( b3MetalPairCandidate );
		uint32_t availableCandidateCount = availableCandidateCount64 < UINT32_MAX
			? (uint32_t)availableCandidateCount64 : UINT32_MAX;
		struct
		{
			uint32_t moveCount, candidateCapacity, candidateLimit, padding;
		} prefixParams = { (uint32_t)moveCount, availableCandidateCount, candidateLimit, 0 };
		_Static_assert( sizeof( prefixParams ) == 16, "Metal pair-prefix parameter ABI changed" );

		id<MTLComputePipelineState> pairPipeline = context->pairCandidatesPipeline;
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil ) return false;
		[encoder setComputePipelineState:pairPipeline];
		[encoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->pairRecordBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->pairCandidateBuffer offset:0 atIndex:3];
		[encoder setBytes:&params length:sizeof( params ) atIndex:4];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairScanBlocksPipeline];
		[encoder setBuffer:context->pairRecordBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairBlockBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreadgroups:MTLSizeMake( pairBlockCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( 256, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairPrefixPipeline];
		[encoder setBuffer:context->pairBlockBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairAddOffsetsPipeline];
		[encoder setBuffer:context->pairRecordBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairBlockBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairAddOffsetsPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		params.writeCandidates = 1;
		[encoder setComputePipelineState:pairPipeline];
		[encoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->pairRecordBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->pairCandidateBuffer offset:0 atIndex:3];
		[encoder setBytes:&params length:sizeof( params ) atIndex:4];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted ) return false;
		if ( stats != NULL ) stats->commandBufferCount = 1;

		double gpuMilliseconds = commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime
			? 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime ) : 0.0;
		b3MetalPairSummary* summary = context->pairSummaryBuffer.contents;
		if ( summary->flags == 4u )
		{
			if ( summary->totalCount > candidateLimit ||
				summary->totalCount > NSUIntegerMax / sizeof( b3MetalPairCandidate ) )
			{
				return false;
			}
			NSUInteger candidateBytes = (NSUInteger)summary->totalCount * sizeof( b3MetalPairCandidate );
			if ( b3MetalEnsurePairCapacity( context, moveBytes, treeBytes, movedBytes, shapeBytes, setBytes, recordBytes, candidateBytes,
				 blockBytes ) == false ) return false;
			summary = context->pairSummaryBuffer.contents;
			summary->flags = 0;
			summary->writeFlags = 0;
			id<MTLCommandBuffer> retryCommand = [context->queue commandBuffer];
			id<MTLComputeCommandEncoder> retryEncoder = [retryCommand computeCommandEncoder];
			if ( retryCommand == nil || retryEncoder == nil ) return false;
			[retryEncoder setComputePipelineState:pairPipeline];
			[retryEncoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
			[retryEncoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
			[retryEncoder setBuffer:context->pairRecordBuffer offset:0 atIndex:2];
			[retryEncoder setBuffer:context->pairCandidateBuffer offset:0 atIndex:3];
			[retryEncoder setBytes:&params length:sizeof( params ) atIndex:4];
			[retryEncoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
			[retryEncoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
			[retryEncoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
			[retryEncoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
			[retryEncoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
			[retryEncoder endEncoding];
			[retryCommand commit];
			[retryCommand waitUntilCompleted];
			if ( retryCommand.status != MTLCommandBufferStatusCompleted ) return false;
			if ( stats != NULL ) stats->commandBufferCount = 2;
			if ( retryCommand.GPUEndTime >= retryCommand.GPUStartTime )
			{
				gpuMilliseconds += 1000.0 * ( retryCommand.GPUEndTime - retryCommand.GPUStartTime );
			}
		}
		if ( summary->flags != 0 || summary->writeFlags != 0 || summary->totalCount > INT32_MAX ) return false;

		*recordsOut = context->pairRecordBuffer.contents;
		*candidatesOut = context->pairCandidateBuffer.contents;
		*candidateCountOut = (int)summary->totalCount;
		if ( stats != NULL ) stats->gpuMilliseconds = gpuMilliseconds;
		return true;
	}
}

static bool b3MetalSupportsHull( const b3Shape* shape )
{
	if ( shape->type != b3_hullShape || shape->hull == NULL ) return false;
	const b3HullData* hull = shape->hull;
	if ( hull->version != B3_HULL_VERSION || hull->vertexCount < 4 || hull->faceCount < 4 || hull->edgeCount < 12 ||
		 b3GetHullPoints( hull ) == NULL || b3GetHullPlanes( hull ) == NULL || b3GetHullFaces( hull ) == NULL ||
		 b3GetHullEdges( hull ) == NULL )
	{
		return false;
	}
	b3Vec3 extent = b3Sub( hull->aabb.upperBound, hull->aabb.lowerBound );
	float minExtent = b3MinFloat( extent.x, b3MinFloat( extent.y, extent.z ) );
	float maxExtent = b3MaxFloat( extent.x, b3MaxFloat( extent.y, extent.z ) );
	// Boundary-triangle closest points are stable for ordinary compact hulls.
	// High-aspect hulls retain GJK on CPU until that exact simplex path is ported.
	return minExtent > B3_LINEAR_SLOP && maxExtent <= 16.0f * minExtent;
}

static bool b3MetalSupportsHullSphere( const b3Shape* shapeA, const b3Shape* shapeB )
{
	return shapeB->type == b3_sphereShape && b3MetalSupportsHull( shapeA );
}

static void b3MetalPackShapeMaterial( b3MetalShapeGeometry* record, const b3Shape* shape )
{
	const b3SurfaceMaterial* material = b3GetShapeMaterials( shape );
	record->friction = material[0].friction;
	record->restitution = material[0].restitution;
	record->rollingResistance = material[0].rollingResistance;
	record->tangentVelocityX = material[0].tangentVelocity.x;
	record->tangentVelocityY = material[0].tangentVelocity.y;
	record->tangentVelocityZ = material[0].tangentVelocity.z;
	if ( shape->type == b3_sphereShape ) record->rollingRadius = shape->sphere.radius;
	else if ( shape->type == b3_capsuleShape ) record->rollingRadius = shape->capsule.radius;
	else if ( shape->type == b3_hullShape ) record->rollingRadius = 0.25f * shape->hull->innerRadius;
}

static bool b3MetalEnsureShapeGeometryRegistry( b3MetalContext* context, const b3World* world )
{
	int shapeCount = world->shapes.count;
	if ( context->convexShapeGeometryRevision == world->metalPairShapeRevision &&
		 context->convexShapeMaterialRevision == world->metalContactMaterialRevision &&
		 context->convexShapeGeometryCount == shapeCount )
	{
		( (b3World*)world )->metalNarrowPhaseGeometryReuseCount += 1;
		return true;
	}

	size_t temporaryCount = shapeCount > 0 ? (size_t)shapeCount : 1;
	int* shapeUniqueIndices = malloc( temporaryCount * sizeof( int ) );
	const b3HullData** uniqueHulls = malloc( temporaryCount * sizeof( const b3HullData* ) );
	b3MetalShapeGeometry* uniqueRecords = calloc( temporaryCount, sizeof( b3MetalShapeGeometry ) );
	if ( shapeUniqueIndices == NULL || uniqueHulls == NULL || uniqueRecords == NULL )
	{
		free( shapeUniqueIndices );
		free( uniqueHulls );
		free( uniqueRecords );
		return false;
	}
	for ( int i = 0; i < shapeCount; ++i ) shapeUniqueIndices[i] = B3_NULL_INDEX;

	b3HullMap hullMap;
	b3HullMap_init( &hullMap );
	b3HullMap_reserve( &hullMap, shapeCount );
	int uniqueCount = 0;
	int supportedShapeCount = 0;
	uint64_t hullPointCount = 0;
	uint64_t hullPlaneCount = 0;
	uint64_t hullTriangleCount = 0;
	bool valid = true;
	for ( int shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex )
	{
		const b3Shape* shape = world->shapes.data + shapeIndex;
		if ( shape->id != shapeIndex || b3MetalSupportsHull( shape ) == false ) continue;
		supportedShapeCount += 1;
		const b3HullData* hull = shape->hull;
		b3HullMap_itr itr = b3HullMap_get_or_insert( &hullMap, hull, uniqueCount );
		int uniqueIndex = itr.data->val;
		shapeUniqueIndices[shapeIndex] = uniqueIndex;
		if ( uniqueIndex != uniqueCount ) continue;
		int triangleCount = hull->edgeCount - 2 * hull->faceCount;
		if ( triangleCount <= 0 )
		{
			valid = false;
			break;
		}
		uniqueHulls[uniqueCount++] = hull;
		hullPointCount += (uint32_t)hull->vertexCount;
		hullPlaneCount += (uint32_t)hull->faceCount;
		hullTriangleCount += (uint32_t)triangleCount;
		if ( hullPointCount > UINT32_MAX || hullPlaneCount > UINT32_MAX || hullTriangleCount > UINT32_MAX )
		{
			valid = false;
			break;
		}
	}
	b3HullMap_cleanup( &hullMap );

	if ( valid && ( hullPointCount > NSUIntegerMax / sizeof( b3MetalFloat4 ) ||
					 hullPlaneCount > NSUIntegerMax / sizeof( b3MetalFloat4 ) ||
					 hullTriangleCount > NSUIntegerMax / sizeof( b3MetalHullTriangle ) ) )
	{
		valid = false;
	}
	NSUInteger hullPointBytes = (NSUInteger)hullPointCount * sizeof( b3MetalFloat4 );
	NSUInteger hullPlaneBytes = (NSUInteger)hullPlaneCount * sizeof( b3MetalFloat4 );
	NSUInteger hullTriangleBytes = (NSUInteger)hullTriangleCount * sizeof( b3MetalHullTriangle );
	if ( valid == false || b3MetalEnsureShapeGeometryCapacity( context, shapeCount, hullPointBytes, hullPlaneBytes,
			hullTriangleBytes ) == false )
	{
		free( shapeUniqueIndices );
		free( uniqueHulls );
		free( uniqueRecords );
		context->convexShapeGeometryRevision = UINT64_MAX;
		context->convexShapeMaterialRevision = UINT64_MAX;
		context->convexShapeGeometryCount = 0;
		return false;
	}

	b3MetalFloat4* hullPoints = context->convexHullPointBuffer.contents;
	b3MetalFloat4* hullPlanes = context->convexHullPlaneBuffer.contents;
	b3MetalHullTriangle* hullTriangles = context->convexHullTriangleBuffer.contents;
	uint32_t hullPointCursor = 0;
	uint32_t hullPlaneCursor = 0;
	uint32_t hullTriangleCursor = 0;
	for ( int uniqueIndex = 0; uniqueIndex < uniqueCount; ++uniqueIndex )
	{
		const b3HullData* hull = uniqueHulls[uniqueIndex];
		const b3Vec3* points = b3GetHullPoints( hull );
		const b3Plane* planes = b3GetHullPlanes( hull );
		const b3HullFace* faces = b3GetHullFaces( hull );
		const b3HullHalfEdge* edges = b3GetHullEdges( hull );
		b3MetalShapeGeometry* record = uniqueRecords + uniqueIndex;
		record->type = b3_hullShape;
		record->pointOffset = hullPointCursor;
		record->pointCount = (uint32_t)hull->vertexCount;
		for ( int pointIndex = 0; pointIndex < hull->vertexCount; ++pointIndex )
		{
			hullPoints[hullPointCursor++] =
				(b3MetalFloat4){ points[pointIndex].x, points[pointIndex].y, points[pointIndex].z, 0.0f };
		}
		record->planeOffset = hullPlaneCursor;
		record->planeCount = (uint32_t)hull->faceCount;
		for ( int planeIndex = 0; planeIndex < hull->faceCount; ++planeIndex )
		{
			hullPlanes[hullPlaneCursor++] = (b3MetalFloat4){ planes[planeIndex].normal.x, planes[planeIndex].normal.y,
				planes[planeIndex].normal.z, planes[planeIndex].offset };
		}
		record->triangleOffset = hullTriangleCursor;
		for ( int faceIndex = 0; faceIndex < hull->faceCount; ++faceIndex )
		{
			uint8_t startEdge = faces[faceIndex].edge;
			uint8_t edgeIndex = edges[startEdge].next;
			uint32_t index1 = edges[startEdge].origin;
			uint32_t index2 = edges[edgeIndex].origin;
			edgeIndex = edges[edgeIndex].next;
			while ( edgeIndex != startEdge )
			{
				uint32_t index3 = edges[edgeIndex].origin;
				hullTriangles[hullTriangleCursor++] =
					(b3MetalHullTriangle){ index1, index2, index3, (uint32_t)faceIndex };
				index2 = index3;
				edgeIndex = edges[edgeIndex].next;
			}
		}
		record->triangleCount = hullTriangleCursor - record->triangleOffset;
		record->supported = 1;
	}

	b3MetalShapeGeometry* records = context->convexShapeGeometryBuffer.contents;
	memset( records, 0, (size_t)shapeCount * sizeof( b3MetalShapeGeometry ) );
	for ( int shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex )
	{
		const b3Shape* shape = world->shapes.data + shapeIndex;
		if ( shape->id != shapeIndex ) continue;
		if ( shape->type == b3_sphereShape )
		{
			records[shapeIndex] = (b3MetalShapeGeometry){
				.point1X = shape->sphere.center.x,
				.point1Y = shape->sphere.center.y,
				.point1Z = shape->sphere.center.z,
				.radius = shape->sphere.radius,
				.point2X = shape->sphere.center.x,
				.point2Y = shape->sphere.center.y,
				.point2Z = shape->sphere.center.z,
				.bodyId = shape->bodyId,
				.type = b3_sphereShape,
				.supported = 1,
			};
			b3MetalPackShapeMaterial( records + shapeIndex, shape );
			continue;
		}
		if ( shape->type == b3_capsuleShape )
		{
			records[shapeIndex] = (b3MetalShapeGeometry){
				.point1X = shape->capsule.center1.x,
				.point1Y = shape->capsule.center1.y,
				.point1Z = shape->capsule.center1.z,
				.radius = shape->capsule.radius,
				.point2X = shape->capsule.center2.x,
				.point2Y = shape->capsule.center2.y,
				.point2Z = shape->capsule.center2.z,
				.bodyId = shape->bodyId,
				.type = b3_capsuleShape,
				.supported = 1,
			};
			b3MetalPackShapeMaterial( records + shapeIndex, shape );
			continue;
		}
		int uniqueIndex = shapeUniqueIndices[shapeIndex];
		if ( uniqueIndex != B3_NULL_INDEX )
		{
			records[shapeIndex] = uniqueRecords[uniqueIndex];
			records[shapeIndex].bodyId = shape->bodyId;
			b3MetalPackShapeMaterial( records + shapeIndex, shape );
		}
	}
	B3_ASSERT( hullPointCursor == hullPointCount );
	B3_ASSERT( hullPlaneCursor == hullPlaneCount );
	B3_ASSERT( hullTriangleCursor == hullTriangleCount );
	context->convexShapeGeometryCount = shapeCount;
	context->convexShapeGeometryRevision = world->metalPairShapeRevision;
	context->convexShapeMaterialRevision = world->metalContactMaterialRevision;
	( (b3World*)world )->metalNarrowPhaseGeometryUploadCount += 1;
	( (b3World*)world )->metalLastNarrowPhaseHullShapeCount = supportedShapeCount;
	( (b3World*)world )->metalLastNarrowPhaseUniqueHullCount = uniqueCount;
	free( shapeUniqueIndices );
	free( uniqueHulls );
	free( uniqueRecords );
	return true;
}

static bool b3MetalEnsureBodyTransformRegistry( b3MetalContext* context, const b3World* world )
{
	int bodyCount = world->bodies.count;
	if ( context->convexBodyTransformStepIndex == world->stepIndex &&
		 context->convexBodyTransformRevision == world->metalBodyTransformRevision &&
		 context->convexBodyTransformCount == bodyCount )
	{
		( (b3World*)world )->metalNarrowPhaseTransformReuseCount += 1;
		return true;
	}
	if ( bodyCount < 0 || (NSUInteger)bodyCount > NSUIntegerMax / sizeof( b3MetalBodyTransform ) ) return false;
	NSUInteger bytes = (NSUInteger)bodyCount * sizeof( b3MetalBodyTransform );
	bytes = bytes > sizeof( b3MetalBodyTransform ) ? bytes : sizeof( b3MetalBodyTransform );
	if ( context->convexBodyTransformCapacity < bytes )
	{
		NSUInteger capacity = context->convexBodyTransformCapacity > 0 ? context->convexBodyTransformCapacity : 4096;
		while ( capacity < bytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexBodyTransformBuffer release];
		context->convexBodyTransformBuffer = buffer;
		context->convexBodyTransformCapacity = capacity;
	}
	b3MetalBodyTransform* transforms = context->convexBodyTransformBuffer.contents;
	memset( transforms, 0, (size_t)bodyCount * sizeof( b3MetalBodyTransform ) );
	for ( int bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex )
	{
		const b3Body* body = world->bodies.data + bodyIndex;
		if ( body->id != bodyIndex || body->setIndex == B3_NULL_INDEX ) continue;
		b3WorldTransform transform = b3GetBodyTransformQuick( (b3World*)world, (b3Body*)body );
		const b3BodySim* bodySim = b3GetBodySim( (b3World*)world, (b3Body*)body );
		b3MetalBodyTransform* output = transforms + bodyIndex;
		output->qx = transform.q.v.x;
		output->qy = transform.q.v.y;
		output->qz = transform.q.v.z;
		output->qw = transform.q.s;
		output->px = (float)transform.p.x;
		output->py = (float)transform.p.y;
		output->pz = (float)transform.p.z;
		output->supported = 1;
		output->localCenterX = bodySim->localCenter.x;
		output->localCenterY = bodySim->localCenter.y;
		output->localCenterZ = bodySim->localCenter.z;
		output->index = body->type == b3_staticBody ? B3_NULL_INDEX : body->localIndex;
		output->flags = bodySim->flags;
#if defined( BOX3D_DOUBLE_PRECISION )
		memcpy( &output->pxBits, &transform.p.x, sizeof( uint64_t ) );
		memcpy( &output->pyBits, &transform.p.y, sizeof( uint64_t ) );
		memcpy( &output->pzBits, &transform.p.z, sizeof( uint64_t ) );
#endif
	}
	context->convexBodyTransformCount = bodyCount;
	context->convexBodyTransformStepIndex = world->stepIndex;
	context->convexBodyTransformRevision = world->metalBodyTransformRevision;
	( (b3World*)world )->metalNarrowPhaseTransformUploadCount += 1;
	return true;
}

static bool b3MetalPrepareBodyTransformDeviceRefresh( b3MetalContext* context, const b3World* world )
{
	if ( context == NULL || world == NULL ) return false;
	int bodyCount = world->bodies.count;
	if ( context->convexBodyTransformBuffer != nil && context->convexBodyTransformCount == bodyCount &&
		 context->convexBodyTransformRevision == world->metalBodyTransformRevision )
	{
		return true;
	}
	return b3MetalEnsureBodyTransformRegistry( context, world );
}

static void b3MetalCommitBodyTransformDeviceRefresh( b3MetalContext* context, b3World* world, int bodyCount )
{
	B3_ASSERT( context != NULL && world != NULL );
	context->convexBodyTransformCount = world->bodies.count;
	context->convexBodyTransformStepIndex = world->stepIndex;
	context->convexBodyTransformRevision = world->metalBodyTransformRevision;
	context->bodyStateResidentCount = bodyCount;
	world->metalNarrowPhaseTransformDeviceRefreshCount += 1;
}

bool b3MetalReadResidentBodyTransform( const b3MetalContext* context, const b3World* world, int bodyId,
	b3WorldTransform* transform, int* bodySimIndex, uint32_t* stateFlags )
{
	if ( context == NULL || world == NULL || transform == NULL || bodyId < 0 || bodyId >= world->bodies.count ||
		 context->convexBodyTransformBuffer == nil || context->convexBodyTransformCount != world->bodies.count ||
		 context->convexBodyTransformStepIndex != world->stepIndex ||
		 context->convexBodyTransformRevision != world->metalBodyTransformRevision )
	{
		return false;
	}
	const b3MetalBodyTransform* record =
		( (const b3MetalBodyTransform*)context->convexBodyTransformBuffer.contents ) + bodyId;
	if ( record->supported == 0 ) return false;
	transform->q = (b3Quat){ { record->qx, record->qy, record->qz }, record->qw };
#if defined( BOX3D_DOUBLE_PRECISION )
	memcpy( &transform->p.x, &record->pxBits, sizeof( uint64_t ) );
	memcpy( &transform->p.y, &record->pyBits, sizeof( uint64_t ) );
	memcpy( &transform->p.z, &record->pzBits, sizeof( uint64_t ) );
#else
	transform->p = (b3Pos){ record->px, record->py, record->pz };
#endif
	if ( bodySimIndex != NULL ) *bodySimIndex = record->index;
	if ( stateFlags != NULL ) *stateFlags = record->flags;
	return true;
}

bool b3MetalSyncBodyMoveEvents( b3MetalContext* context, b3World* world, b3BodyMoveEvent* events, int eventCount )
{
	if ( context == NULL || world == NULL || events == NULL || eventCount < 0 ||
		 context->bodyMoveResultBuffer == nil || context->bodyMoveReadbackBuffer == nil ||
		 context->bodyMoveResultCount != eventCount || context->bodyMoveResultStepIndex != world->stepIndex )
	{
		return false;
	}
	if ( (NSUInteger)eventCount > NSUIntegerMax / sizeof( b3MetalBodyMoveResult ) ) return false;
	NSUInteger bytes = (NSUInteger)eventCount * sizeof( b3MetalBodyMoveResult );
	if ( bytes > context->bodyMoveResultCapacity || bytes > context->bodyMoveReadbackCapacity ) return false;

	@autoreleasepool
	{
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
		if ( commandBuffer == nil || blit == nil ) return false;
		[blit copyFromBuffer:context->bodyMoveResultBuffer sourceOffset:0 toBuffer:context->bodyMoveReadbackBuffer
			destinationOffset:0 size:bytes];
		[blit endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted ) return false;
	}

	const b3MetalBodyMoveResult* results = context->bodyMoveReadbackBuffer.contents;
	for ( int i = 0; i < eventCount; ++i )
	{
		int bodyId = results[i].bodyId;
		if ( bodyId < 0 || bodyId >= world->bodies.count ) return false;
		uint16_t worldId = (uint16_t)( results[i].generationWorld & 0xffffu );
		uint16_t generation = (uint16_t)( results[i].generationWorld >> 16 );
		const b3Body* body = world->bodies.data + bodyId;
		if ( body->id != bodyId || worldId != world->worldId || generation != body->generation ) return false;
	}
	for ( int i = 0; i < eventCount; ++i )
	{
		const b3MetalBodyMoveResult* result = results + i;
		b3BodyMoveEvent* event = events + i;
		event->userData = (void*)(uintptr_t)result->userData;
		event->transform.q = (b3Quat){ { result->qx, result->qy, result->qz }, result->qw };
#if defined( BOX3D_DOUBLE_PRECISION )
		memcpy( &event->transform.p.x, &result->pxBits, sizeof( uint64_t ) );
		memcpy( &event->transform.p.y, &result->pyBits, sizeof( uint64_t ) );
		memcpy( &event->transform.p.z, &result->pzBits, sizeof( uint64_t ) );
#else
		event->transform.p = (b3Pos){ result->px, result->py, result->pz };
#endif
		event->bodyId = (b3BodyId){ result->bodyId + 1, (uint16_t)( result->generationWorld & 0xffffu ),
			(uint16_t)( result->generationWorld >> 16 ) };
		event->fellAsleep = false;
	}
	world->metalBodyMoveEventSyncCount += 1;
	world->metalLastBodyMoveEventReadbackBytes = bytes;
	return true;
}

bool b3MetalCanReuseConvexManifoldInputs( const b3MetalContext* context, const b3World* world, int contactCount )
{
	return context != NULL && world != NULL && contactCount > 0 && context->convexManifoldInputBuffer != nil &&
		context->convexManifoldInputCount == contactCount &&
		context->convexManifoldInputPairRevision == world->broadPhase.pairSetRevision &&
		context->convexManifoldInputGraphRevision == world->constraintGraph.revision &&
		context->convexManifoldInputRevision == world->metalContactInputRevision;
}

bool b3MetalComputeConvexManifolds( b3MetalContext* context, const b3World* world, const int* contactIndices,
	int contactCount, const b3MetalConvexManifoldResult** resultsOut, int* resultCountOut, int* residentBypassCountOut,
	b3MetalDispatchStats* stats )
{
	if ( resultsOut != NULL ) *resultsOut = NULL;
	if ( resultCountOut != NULL ) *resultCountOut = 0;
	if ( residentBypassCountOut != NULL ) *residentBypassCountOut = 0;
	if ( stats != NULL ) *stats = (b3MetalDispatchStats){ .bodyCount = contactCount };
	if ( context == NULL || world == NULL || contactCount < 0 || resultsOut == NULL ||
		 resultCountOut == NULL )
	{
		return false;
	}
	bool reuseInputs = contactIndices == NULL;
	if ( reuseInputs && b3MetalCanReuseConvexManifoldInputs( context, world, contactCount ) == false ) return false;
	( (b3World*)world )->metalLastNarrowPhaseResultCount = 0;
	( (b3World*)world )->metalLastNarrowPhaseManifoldTableCount = 0;
	context->convexManifoldTableCount = 0;
	if ( contactCount == 0 ) return true;
	if ( (NSUInteger)contactCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldInput ) ||
		 (NSUInteger)contactCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) ||
		 world->contacts.count < 0 || (NSUInteger)world->contacts.count > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) )
	{
		return false;
	}

	@autoreleasepool
	{
		int candidateCount = reuseInputs ? context->convexManifoldCandidateCount : 0;
		int hitEventContactCount = reuseInputs ? context->contactHitEventIdCount : 0;
		if ( reuseInputs == false )
		{
			for ( int i = 0; i < contactCount; ++i )
			{
				int contactIndex = contactIndices[i];
				if ( contactIndex < 0 || contactIndex >= world->contacts.count ) return false;
				const b3Contact* contact = world->contacts.data + contactIndex;
				if ( contact->contactId != contactIndex || contact->shapeIdA < 0 || contact->shapeIdA >= world->shapes.count ||
					 contact->shapeIdB < 0 || contact->shapeIdB >= world->shapes.count )
				{
					return false;
				}
				const b3Shape* shapeA = world->shapes.data + contact->shapeIdA;
				const b3Shape* shapeB = world->shapes.data + contact->shapeIdB;
				bool eligible = ( shapeA->type == b3_sphereShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_capsuleShape ) ||
					b3MetalSupportsHullSphere( shapeA, shapeB );
				candidateCount += eligible;
				hitEventContactCount += eligible && ( ( shapeA->flags & b3_enableHitEvents ) != 0 ||
					( shapeB->flags & b3_enableHitEvents ) != 0 );
			}
		}
		if ( reuseInputs == false ) context->contactHitEventIdCount = 0;
		if ( candidateCount == 0 ) return true;

		NSUInteger inputBytes = (NSUInteger)contactCount * sizeof( b3MetalConvexManifoldInput );
		NSUInteger resultBytes = (NSUInteger)contactCount * sizeof( b3MetalConvexManifoldResult );
		uint32_t blockCount = ( (uint32_t)contactCount + 255u ) / 256u;
		NSUInteger blockBytes = (NSUInteger)blockCount * sizeof( b3MetalPairBlock );
		NSUInteger tableCount = world->contacts.count > 0 ? (NSUInteger)world->contacts.count : 1;
		NSUInteger tableBytes = tableCount * sizeof( b3MetalConvexManifoldResult );
		if ( tableCount > NSUIntegerMax / sizeof( b3MetalContactPrepareInput ) ||
			 tableCount > NSUIntegerMax / sizeof( b3MetalContactImpulseResult ) ) return false;
		NSUInteger prepareTableBytes = tableCount * sizeof( b3MetalContactPrepareInput );
		NSUInteger impulseTableBytes = tableCount * sizeof( b3MetalContactImpulseResult );
		NSUInteger previousImpulseCapacity = context->contactImpulseResultCapacity;
		if ( b3MetalEnsureShapeGeometryRegistry( context, world ) == false ||
			 b3MetalEnsureBodyTransformRegistry( context, world ) == false ||
			 b3MetalEnsureConvexManifoldCapacity( context, inputBytes, resultBytes, resultBytes, blockBytes, tableBytes ) == false ||
			 b3MetalEnsureContactPrepareTableCapacity( context, prepareTableBytes ) == false ||
			 b3MetalEnsureContactImpulseResultCapacity( context, impulseTableBytes ) == false ||
			 b3MetalEnsureContactHitEventIdCapacity( context, (NSUInteger)hitEventContactCount * sizeof( int ) ) == false )
		{
			return false;
		}
		if ( context->contactImpulseResultCapacity != previousImpulseCapacity )
		{
			// Growing the table replaces its storage, so prior warm starts are no longer resident.
			context->contactImpulseResultCount = 0;
		}
		context->contactPrepareGeneration += 1;
		if ( context->contactPrepareGeneration == 0 )
		{
			memset( context->contactPrepareTableBuffer.contents, 0, prepareTableBytes );
			context->contactPrepareGeneration = 1;
		}
		b3MetalConvexManifoldInput* inputs = context->convexManifoldInputBuffer.contents;
		if ( reuseInputs == false )
		{
			int* hitEventContactIds = context->contactHitEventIdBuffer.contents;
			memset( inputs, 0, inputBytes );
			for ( int i = 0; i < contactCount; ++i )
			{
				int contactIndex = contactIndices[i];
				if ( contactIndex < 0 || contactIndex >= world->contacts.count ) return false;
				const b3Contact* contact = world->contacts.data + contactIndex;
				if ( contact->contactId != contactIndex || contact->shapeIdA < 0 ||
					 contact->shapeIdA >= world->shapes.count || contact->shapeIdB < 0 ||
					 contact->shapeIdB >= world->shapes.count )
				{
					return false;
				}
				const b3Shape* shapeA = world->shapes.data + contact->shapeIdA;
				const b3Shape* shapeB = world->shapes.data + contact->shapeIdB;
				bool eligible = ( shapeA->type == b3_sphereShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_capsuleShape ) ||
					b3MetalSupportsHullSphere( shapeA, shapeB );
				if ( eligible == false ) continue;
				if ( ( shapeA->flags & b3_enableHitEvents ) != 0 || ( shapeB->flags & b3_enableHitEvents ) != 0 )
				{
					hitEventContactIds[context->contactHitEventIdCount++] = contactIndex;
				}
				b3MetalConvexManifoldInput* input = inputs + i;
				input->eligible = 1;
				input->shapeIdA = (uint32_t)contact->shapeIdA;
				input->shapeIdB = (uint32_t)contact->shapeIdB;
				input->contactId = (uint32_t)contactIndex;
				input->contactGeneration = contact->generation;
				bool prepareEligible = world->contactRecycleDistance == 0.0f &&
					( contact->flags & b3_simEnablePreSolveEvents ) == 0 && world->metalDefaultFrictionCallback &&
					world->metalDefaultRestitutionCallback;
				input->prepareEligible = prepareEligible ? 1u : 0u;
				if ( contact->shapeIdA >= context->convexShapeGeometryCount ||
					 contact->shapeIdB >= context->convexShapeGeometryCount )
				{
					return false;
				}
				const b3MetalShapeGeometry* geometry = context->convexShapeGeometryBuffer.contents;
				if ( geometry[contact->shapeIdA].supported == 0 || geometry[contact->shapeIdB].supported == 0 ) return false;
				int bodyIdA = geometry[contact->shapeIdA].bodyId;
				int bodyIdB = geometry[contact->shapeIdB].bodyId;
				if ( bodyIdA < 0 || bodyIdA >= context->convexBodyTransformCount || bodyIdB < 0 ||
					 bodyIdB >= context->convexBodyTransformCount ) return false;
				const b3MetalBodyTransform* transforms = context->convexBodyTransformBuffer.contents;
				if ( transforms[bodyIdA].supported == 0 || transforms[bodyIdB].supported == 0 ) return false;
				const b3Body* bodyA = world->bodies.data + bodyIdA;
				const b3Body* bodyB = world->bodies.data + bodyIdB;
				input->indexA = bodyA->type == b3_staticBody ? B3_NULL_INDEX : bodyA->localIndex;
				input->indexB = bodyB->type == b3_staticBody ? B3_NULL_INDEX : bodyB->localIndex;
				const b3BodySim* simA = b3GetBodySim( (b3World*)world, (b3Body*)bodyA );
				const b3BodySim* simB = b3GetBodySim( (b3World*)world, (b3Body*)bodyB );
				bool isFast = ( simA->flags & b3_isFast ) != 0 || ( simB->flags & b3_isFast ) != 0;
				bool bypassEligible = prepareEligible && ( contact->flags & b3_simTouchingFlag ) != 0 &&
					( contact->flags & b3_simMetalManifold ) != 0 && contact->manifoldCount == 1 &&
					contact->manifolds != NULL && isFast == false && ( shapeA->flags & b3_enableHitEvents ) == 0 &&
					( shapeB->flags & b3_enableHitEvents ) == 0 && world->recording == NULL;
				input->prepareEligible |= bypassEligible ? 2u : 0u;
			}
			context->convexManifoldInputCount = contactCount;
			context->convexManifoldCandidateCount = candidateCount;
			context->convexManifoldInputPairRevision = world->broadPhase.pairSetRevision;
			context->convexManifoldInputGraphRevision = world->constraintGraph.revision;
			context->convexManifoldInputRevision = world->metalContactInputRevision;
			( (b3World*)world )->metalContactInputPackCount += 1;
			( (b3World*)world )->metalLastContactInputBytes = inputBytes;
		}
		else
		{
			( (b3World*)world )->metalContactInputReuseCount += 1;
			( (b3World*)world )->metalLastContactInputBytes = 0;
		}

		struct { uint32_t contactCount; float linearSlop, speculativeDistance; uint32_t bodyCount; } params = {
			(uint32_t)contactCount, B3_LINEAR_SLOP, B3_SPECULATIVE_DISTANCE, (uint32_t)context->convexBodyTransformCount,
		};
		struct
		{
			uint32_t contactCount, blockCount, previousCount, previousGeneration;
			uint32_t currentGeneration, padding0, padding1, padding2;
		} compactParams = {
			(uint32_t)contactCount, blockCount, (uint32_t)context->contactImpulseResultCount,
			context->contactImpulseResultGeneration, context->contactPrepareGeneration,
			residentBypassCountOut != NULL ? 1u : 0u, 0u, 0u,
		};
		NSUInteger scanWidth = context->convexManifoldScanPipeline.threadExecutionWidth;
		if ( context->convexManifoldScanPipeline.maxTotalThreadsPerThreadgroup < 256 || scanWidth == 0 ||
			 256 % scanWidth != 0 || 256 / scanWidth > 32 )
		{
			return false;
		}
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil ) return false;
		id<MTLComputePipelineState> pipeline = context->convexManifoldPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->convexManifoldInputBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldResultBuffer offset:0 atIndex:1];
		[encoder setBytes:&params length:sizeof( params ) atIndex:2];
		[encoder setBuffer:context->convexHullPointBuffer offset:0 atIndex:3];
		[encoder setBuffer:context->convexHullPlaneBuffer offset:0 atIndex:4];
		[encoder setBuffer:context->convexHullTriangleBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->convexShapeGeometryBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->convexBodyTransformBuffer offset:0 atIndex:7];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)contactCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

		pipeline = context->convexManifoldScanPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->convexManifoldResultBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldBlockBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->convexManifoldInputBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->contactImpulseResultBuffer offset:0 atIndex:3];
		[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:4];
		[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:5];
		[encoder dispatchThreadgroups:MTLSizeMake( blockCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 256, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

		pipeline = context->convexManifoldPrefixPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->convexManifoldBlockBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldSummaryBuffer offset:0 atIndex:1];
		[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:2];
		[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

		pipeline = context->convexManifoldScatterPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->convexManifoldResultBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldBlockBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->convexManifoldCompactBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->convexManifoldInputBuffer offset:0 atIndex:3];
		[encoder setBuffer:context->convexShapeGeometryBuffer offset:0 atIndex:4];
		[encoder setBuffer:context->convexBodyTransformBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->convexManifoldTableBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->contactImpulseResultBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:8];
		[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:9];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)contactCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pipeline ), 1, 1 )];
		[encoder endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted ) return false;
		const b3MetalPairSummary* summary = context->convexManifoldSummaryBuffer.contents;
		uint64_t resultLimit = residentBypassCountOut != NULL ? (uint64_t)contactCount : (uint64_t)candidateCount;
		if ( (uint64_t)summary->writeFlags > 2ull * summary->flags || summary->totalCount > resultLimit ||
			 summary->totalCount > INT32_MAX ||
			 summary->flags > (uint32_t)candidateCount ||
			 ( residentBypassCountOut != NULL && summary->totalCount + summary->flags != (uint64_t)contactCount ) )
		{
			return false;
		}
		int resultCount = (int)summary->totalCount;
		int residentBypassCount = residentBypassCountOut != NULL ? (int)summary->flags : 0;
		const b3MetalConvexManifoldResult* completedResults = context->convexManifoldCompactBuffer.contents;
		uint64_t persistenceMatchCount = summary->writeFlags;
		uint64_t prepareDeviceRefreshCount = (uint64_t)residentBypassCount;
		for ( int i = 0; i < resultCount; ++i )
		{
			uint32_t persistedBits = completedResults[i].persistedBits;
			persistenceMatchCount += ( persistedBits & 1u ) + ( ( persistedBits >> 1 ) & 1u );
			prepareDeviceRefreshCount += ( completedResults[i].residentFlags & 2u ) != 0;
		}
		( (b3World*)world )->metalContactPersistenceMatchCount += persistenceMatchCount;
		( (b3World*)world )->metalContactPrepareDeviceRefreshCount += prepareDeviceRefreshCount;
		context->convexManifoldTableCount = world->contacts.count;
		( (b3World*)world )->metalLastNarrowPhaseResultCount = resultCount;
		( (b3World*)world )->metalLastNarrowPhaseManifoldTableCount = world->contacts.count;
		*resultsOut = completedResults;
		*resultCountOut = resultCount;
		if ( residentBypassCountOut != NULL ) *residentBypassCountOut = residentBypassCount;
		if ( stats != NULL )
		{
			stats->commandBufferCount = 1;
			if ( commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
			{
				stats->gpuMilliseconds = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
			}
		}
		return true;
	}
}

static bool b3MetalReadbackConvexManifoldRange( b3MetalContext* context, NSUInteger sourceOffset, NSUInteger bytes )
{
	if ( context == NULL || context->convexManifoldTableBuffer == nil || bytes == 0 || sourceOffset > NSUIntegerMax - bytes )
		return false;
	@autoreleasepool
	{
		if ( context->convexManifoldTableReadbackCapacity < bytes )
		{
			NSUInteger capacity = context->convexManifoldTableReadbackCapacity > 0 ?
				context->convexManifoldTableReadbackCapacity : 4096;
			while ( capacity < bytes ) capacity *= 2;
			id<MTLBuffer> buffer =
				[context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
			if ( buffer == nil ) return false;
			[context->convexManifoldTableReadbackBuffer release];
			context->convexManifoldTableReadbackBuffer = buffer;
			context->convexManifoldTableReadbackCapacity = capacity;
		}
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
		if ( commandBuffer == nil || blit == nil ) return false;
		[blit copyFromBuffer:context->convexManifoldTableBuffer sourceOffset:sourceOffset
			toBuffer:context->convexManifoldTableReadbackBuffer destinationOffset:0 size:bytes];
		[blit endEncoding];
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted ) return false;
		return true;
	}
}

bool b3MetalCopyResidentConvexManifoldTable( b3MetalContext* context, b3MetalConvexManifoldResult* results,
	int resultCapacity )
{
	if ( context == NULL || resultCapacity < 0 || resultCapacity < context->convexManifoldTableCount ||
		( context->convexManifoldTableCount > 0 && results == NULL ) )
	{
		return false;
	}
	int resultCount = context->convexManifoldTableCount;
	if ( resultCount == 0 ) return true;
	if ( (NSUInteger)resultCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) ) return false;
	NSUInteger bytes = (NSUInteger)resultCount * sizeof( b3MetalConvexManifoldResult );
	if ( b3MetalReadbackConvexManifoldRange( context, 0, bytes ) == false ) return false;
	memcpy( results, context->convexManifoldTableReadbackBuffer.contents, bytes );
	return true;
}

static bool b3MetalApplyContactManifoldResult( b3Contact* contact, const b3MetalConvexManifoldResult* result )
{
	if ( contact == NULL || result == NULL || contact->contactId < 0 || contact->manifoldCount != 1 ||
		contact->manifolds == NULL || result->eligible == 0 || result->touching == 0 || result->pointCount < 1 ||
		result->pointCount > 2 || result->contactId != (uint32_t)contact->contactId ||
		result->inputIndex != (uint32_t)contact->contactId || result->contactGeneration != contact->generation )
	{
		return false;
	}

	b3Manifold* manifold = contact->manifolds;
	manifold->normal = (b3Vec3){ result->normalX, result->normalY, result->normalZ };
	manifold->pointCount = (int)result->pointCount;
	const b3Vec3 anchorAs[2] = {
		{ result->point1X, result->point1Y, result->point1Z },
		{ result->point2X, result->point2Y, result->point2Z },
	};
	const b3Vec3 anchorBs[2] = {
		{ result->anchorB1X, result->anchorB1Y, result->anchorB1Z },
		{ result->anchorB2X, result->anchorB2Y, result->anchorB2Z },
	};
	const float separations[2] = { result->separation1, result->separation2 };
	const float normalImpulses[2] = { result->normalImpulse1, result->normalImpulse2 };
	const uint32_t featureIds[2] = { result->featureId1, result->featureId2 };
	for ( int pointIndex = 0; pointIndex < manifold->pointCount; ++pointIndex )
	{
		b3ManifoldPoint* point = manifold->points + pointIndex;
		point->anchorA = anchorAs[pointIndex];
		point->anchorB = anchorBs[pointIndex];
		point->separation = separations[pointIndex];
		point->baseSeparation = separations[pointIndex];
		point->normalImpulse = normalImpulses[pointIndex];
		point->totalNormalImpulse = 0.0f;
		point->normalVelocity = 0.0f;
		point->featureId = featureIds[pointIndex];
		point->triangleIndex = B3_NULL_INDEX;
		point->persisted = ( result->persistedBits & ( 1u << pointIndex ) ) != 0;
	}
	contact->friction = result->friction;
	contact->restitution = result->restitution;
	contact->rollingResistance = result->rollingResistance;
	contact->tangentVelocity =
		(b3Vec3){ result->tangentVelocityX, result->tangentVelocityY, result->tangentVelocityZ };
	contact->flags &= ~b3_simMetalManifoldStale;
	return true;
}

bool b3MetalSyncContactManifold( b3MetalContext* context, b3Contact* contact )
{
	if ( context == NULL || contact == NULL ) return false;
	if ( contact->contactId < 0 || contact->contactId >= context->convexManifoldTableCount ) return false;
	NSUInteger offset = (NSUInteger)contact->contactId * sizeof( b3MetalConvexManifoldResult );
	if ( b3MetalReadbackConvexManifoldRange( context, offset, sizeof( b3MetalConvexManifoldResult ) ) == false ) return false;
	return b3MetalApplyContactManifoldResult( contact, context->convexManifoldTableReadbackBuffer.contents );
}

bool b3MetalSyncAllContactManifolds( b3MetalContext* context, b3World* world )
{
	if ( world == NULL ) return false;
	int staleCount = 0;
	for ( int contactId = 0; contactId < world->contacts.count; ++contactId )
	{
		const b3Contact* contact = world->contacts.data + contactId;
		staleCount += contact->contactId == contactId && b3IsContactManifoldStale( world, contact );
	}
	if ( staleCount == 0 ) return true;
	if ( context == NULL ) return false;
	if ( context->convexManifoldTableCount <= 0 ||
		(NSUInteger)context->convexManifoldTableCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) ) return false;
	NSUInteger bytes = (NSUInteger)context->convexManifoldTableCount * sizeof( b3MetalConvexManifoldResult );
	if ( b3MetalReadbackConvexManifoldRange( context, 0, bytes ) == false ) return false;
	const b3MetalConvexManifoldResult* results = context->convexManifoldTableReadbackBuffer.contents;
	for ( int contactId = 0; contactId < world->contacts.count; ++contactId )
	{
		b3Contact* contact = world->contacts.data + contactId;
		if ( contact->contactId != contactId || b3IsContactManifoldStale( world, contact ) == false ) continue;
		if ( contactId >= context->convexManifoldTableCount ||
			b3MetalApplyContactManifoldResult( contact, results + contactId ) == false ) return false;
		contact->metalSyncGeneration = world->metalContactManifoldGeneration;
		world->metalContactManifoldSyncCount += 1;
	}
	return true;
}

void b3MetalCommitPairTreeRefit( b3MetalContext* context, const b3BroadPhase* broadPhase )
{
	if ( context == NULL || broadPhase == NULL || context->pairTreeBuffer == nil ) return;
	context->pairTreeRevision = broadPhase->treeRevision;
	context->shapeBoundsRevision = broadPhase->treeRevision;
}

void b3MetalInvalidateShapeInputCache( b3MetalContext* context )
{
	if ( context != NULL ) context->shapeInputCacheValid = false;
}

static bool b3MetalApplyShapeBoundResult( b3MetalContext* context, b3World* world, int shapeId )
{
	if ( context == NULL || world == NULL || context->shapeReadbackBuffer == nil ||
		shapeId < 0 || shapeId >= world->shapes.count )
	{
		return false;
	}

	b3Shape* shape = world->shapes.data + shapeId;
	int resultIndex = shape->metalResultIndex;
	if ( resultIndex < 0 || resultIndex >= context->shapeBoundsCount )
	{
		context->shapeInputCacheValid = false;
		return false;
	}
	if ( shape->metalSyncGeneration == world->metalShapeResultGeneration ) return true;
	const b3MetalShapeAABBResult* results = context->shapeReadbackBuffer.contents;
	const b3MetalShapeAABBResult* result = results + resultIndex;
	if ( result->shapeId != shapeId || shape->id != shapeId )
	{
		context->shapeInputCacheValid = false;
		return false;
	}

	shape->aabb = (b3AABB){
		{ result->lowerX, result->lowerY, result->lowerZ },
		{ result->upperX, result->upperY, result->upperZ },
	};
	shape->fatAABB = (b3AABB){
		{ result->fatLowerX, result->fatLowerY, result->fatLowerZ },
		{ result->fatUpperX, result->fatUpperY, result->fatUpperZ },
	};
	shape->metalSyncGeneration = world->metalShapeResultGeneration;
	world->metalShapeBoundsSyncCount += 1;
	return true;
}

bool b3MetalSyncShapeBounds( b3MetalContext* context, b3World* world, int shapeId )
{
	if ( context == NULL || world == NULL || world->metalShapeCpuBoundsStale == false ||
		shapeId < 0 || shapeId >= world->shapes.count ) return false;
	b3Shape* shape = world->shapes.data + shapeId;
	int resultIndex = shape->metalResultIndex;
	if ( resultIndex < 0 || resultIndex >= context->shapeBoundsCount )
	{
		context->shapeInputCacheValid = false;
		return false;
	}
	if ( shape->metalSyncGeneration == world->metalShapeResultGeneration ) return true;
	NSUInteger offset = (NSUInteger)resultIndex * sizeof( b3MetalShapeAABBResult );
	if ( b3MetalReadbackShapeRange( context, offset, sizeof( b3MetalShapeAABBResult ) ) == false )
	{
		context->shapeInputCacheValid = false;
		return false;
	}
	return b3MetalApplyShapeBoundResult( context, world, shapeId );
}

bool b3MetalSyncAllShapeBounds( b3MetalContext* context, b3World* world )
{
	if ( world == NULL ) return false;
	if ( world->metalShapeCpuBoundsStale == false ) return true;
	if ( context == NULL ) return false;
	int resultCount = context->shapeBoundsCount;
	if ( resultCount <= 0 || b3MetalReadbackShapeRange( context, 0,
		(NSUInteger)resultCount * sizeof( b3MetalShapeAABBResult ) ) == false ) return false;
	const b3MetalShapeAABBResult* results = context->shapeReadbackBuffer.contents;
	for ( int resultIndex = 0; resultIndex < resultCount; ++resultIndex )
	{
		int shapeId = results[resultIndex].shapeId;
		if ( shapeId < 0 || shapeId >= world->shapes.count )
		{
			context->shapeInputCacheValid = false;
			return false;
		}
		b3Shape* shape = world->shapes.data + shapeId;
		if ( shape->metalResultIndex == B3_NULL_INDEX ) continue;
		if ( b3MetalApplyShapeBoundResult( context, world, shapeId ) == false ) return false;
	}
	world->metalShapeCpuBoundsStale = false;
	return true;
}

bool b3MetalSolveContactSubsteps( b3MetalContext* context, b3StepContext* stepContext,
	int velocityIterations, int relaxIterations, int restitutionIterations, b3MetalDispatchStats* stats )
{
	int bodyCount = stepContext != NULL ? stepContext->world->solverSets.data[b3_awakeSet].bodyStates.count : 0;
	if ( stats != NULL )
	{
		*stats = (b3MetalDispatchStats){ .bodyCount = bodyCount };
	}
	if ( context != NULL ) context->contactImpulseResultCount = 0;
	if ( context == NULL || stepContext == NULL || bodyCount < 0 ||
		( stepContext->wideContactCount <= 0 && stepContext->contactConstraintCount <= 0 &&
		  stepContext->overflowContactConstraintCount <= 0 && stepContext->jointConstraintCount <= 0 &&
		  stepContext->overflowJointConstraintCount <= 0 ) ||
		velocityIterations < 1 || relaxIterations < 0 || restitutionIterations < 0 )
	{
		if ( context != NULL ) context->bodyPropertiesResidentCount = 0;
		return false;
	}

	@autoreleasepool
	{
		NSUInteger stateBytes = (NSUInteger)bodyCount * sizeof( b3BodyState );
		NSUInteger propertyBytes = (NSUInteger)bodyCount * sizeof( b3MetalBodyProperties );
		NSUInteger contactBytes = (NSUInteger)stepContext->wideContactCount * sizeof( b3ContactConstraintWide );
		NSUInteger coloredMeshContactBytes =
			(NSUInteger)stepContext->contactConstraintCount * sizeof( b3ContactConstraint );
		NSUInteger overflowMeshContactBytes =
			(NSUInteger)stepContext->overflowContactConstraintCount * sizeof( b3ContactConstraint );
		NSUInteger coloredMeshManifoldBytes =
			(NSUInteger)stepContext->manifoldConstraintCount * sizeof( b3ManifoldConstraint );
		NSUInteger overflowMeshManifoldBytes =
			(NSUInteger)stepContext->overflowManifoldConstraintCount * sizeof( b3ManifoldConstraint );
		NSUInteger meshContactBytes = coloredMeshContactBytes + overflowMeshContactBytes;
		NSUInteger meshManifoldBytes = coloredMeshManifoldBytes + overflowMeshManifoldBytes;
		int distanceOffsets[B3_GRAPH_COLOR_COUNT] = { 0 };
		int distanceCounts[B3_GRAPH_COLOR_COUNT] = { 0 };
		int parallelOffsets[B3_GRAPH_COLOR_COUNT] = { 0 };
		int parallelCounts[B3_GRAPH_COLOR_COUNT] = { 0 };
		int coloredDistanceCount = 0;
		int coloredParallelCount = 0;
		b3GraphColor* overflow = stepContext->graph->colors + B3_OVERFLOW_INDEX;
		for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
		{
			distanceOffsets[colorIndex] = coloredDistanceCount;
			parallelOffsets[colorIndex] = coloredParallelCount;
			int count = stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = stepContext->jointPrepareSpans[colorIndex].joints;
			for ( int i = 0; i < count; ++i )
			{
				if ( joints[i].type == b3_distanceJoint ) distanceCounts[colorIndex] += 1;
				if ( joints[i].type == b3_parallelJoint ) parallelCounts[colorIndex] += 1;
			}
			coloredDistanceCount += distanceCounts[colorIndex];
			coloredParallelCount += parallelCounts[colorIndex];
		}
		int overflowDistanceCount = 0;
		int overflowParallelCount = 0;
		for ( int i = 0; i < stepContext->overflowJointConstraintCount; ++i )
		{
			if ( overflow->jointSims.data[i].type == b3_distanceJoint ) overflowDistanceCount += 1;
			if ( overflow->jointSims.data[i].type == b3_parallelJoint ) overflowParallelCount += 1;
		}
		int distanceCount = coloredDistanceCount + overflowDistanceCount;
		int parallelCount = coloredParallelCount + overflowParallelCount;
		NSUInteger distanceJointBytes = (NSUInteger)distanceCount * sizeof( b3MetalDistanceJoint );
		NSUInteger parallelJointBytes = (NSUInteger)parallelCount * sizeof( b3MetalParallelJoint );
		NSUInteger jointOverflowBytes =
			(NSUInteger)stepContext->overflowJointConstraintCount * sizeof( b3MetalJointOverflow );
		bool prepareContactsOnGpu = stepContext->metalPrepareConvexOnGpu;
		bool preparedHasRestitution = false;
		bool finalizeBodies = stepContext->world->metalFinalizationEnabled;
		bool omitFinalizeReadback = finalizeBodies && stepContext->world->enableSleep == false &&
			stepContext->world->enableContinuous == false;
		bool publishBodyTransforms = omitFinalizeReadback &&
			b3MetalPrepareBodyTransformDeviceRefresh( context, stepContext->world );
		NSUInteger finalizationBytes = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeResult );
		NSUInteger finalizePropertyBytes = (NSUInteger)bodyCount * sizeof( b3MetalFinalizeProperties );
		NSUInteger bodyMoveBytes = (NSUInteger)bodyCount * sizeof( b3MetalBodyMoveResult );
		context->bodyMoveResultCount = 0;
		if ( b3MetalEnsureBodyCapacity( context, stateBytes ) == false ||
			 b3MetalEnsurePropertiesCapacity( context, propertyBytes ) == false ||
			 ( finalizeBodies && ( b3MetalEnsureFinalizeResultCapacity( context, finalizationBytes ) == false ||
			   b3MetalEnsureFinalizePropertiesCapacity( context, finalizePropertyBytes ) == false ||
			   ( publishBodyTransforms && b3MetalEnsureBodyMoveCapacity( context, bodyMoveBytes ) == false ) ) ) ||
			 ( contactBytes > 0 && b3MetalEnsureContactCapacity( context, contactBytes ) == false ) ||
			 ( prepareContactsOnGpu && b3MetalEnsureContactImpulseResultCapacity( context,
			   (NSUInteger)context->convexManifoldTableCount * sizeof( b3MetalContactImpulseResult ) ) == false ) ||
			 ( meshContactBytes > 0 && b3MetalEnsureMeshContactCapacity( context, meshContactBytes ) == false ) ||
			 ( meshManifoldBytes > 0 && b3MetalEnsureMeshManifoldCapacity( context, meshManifoldBytes ) == false ) ||
			 ( distanceJointBytes > 0 && b3MetalEnsureDistanceJointCapacity( context, distanceJointBytes ) == false ) ||
			 ( parallelJointBytes > 0 && b3MetalEnsureParallelJointCapacity( context, parallelJointBytes ) == false ) ||
			 ( jointOverflowBytes > 0 && b3MetalEnsureJointOverflowCapacity( context, jointOverflowBytes ) == false ) )
		{
			context->bodyPropertiesResidentCount = 0;
			return false;
		}

		bool reuseBodyStates = publishBodyTransforms && context->bodyStateResidentCount == bodyCount &&
			context->bodyStateResidentRevision == stepContext->world->metalBodyStateRevision;
		stepContext->world->metalBodyStateRevisionCheckCount += publishBodyTransforms ? 1 : 0;
		bool reuseBodyProperties = publishBodyTransforms && context->bodyPropertiesResidentCount == bodyCount &&
			context->bodyPropertiesResidentRevision == stepContext->world->metalBodyPropertyRevision;
		// A failed command must not leave the previous generation reusable.
		context->bodyStateResidentCount = 0;
		context->bodyPropertiesResidentCount = 0;
		if ( reuseBodyStates == false )
		{
			memcpy( context->bodyStateBuffer.contents, stepContext->states, stateBytes );
		}
		stepContext->world->metalBodyStateReuseCount += reuseBodyStates ? 1 : 0;
		stepContext->world->metalBodyStateUploadCount += reuseBodyStates ? 0 : 1;
		stepContext->world->metalLastBodyStateUploadBytes = reuseBodyStates ? 0 : stateBytes;
		if ( reuseBodyProperties == false )
		{
			b3MetalPackBodyProperties( context->bodyPropertiesBuffer.contents, stepContext->sims, bodyCount );
		}
		stepContext->world->metalBodyPropertyReuseCount += reuseBodyProperties ? 1 : 0;
		stepContext->world->metalBodyPropertyUploadCount += reuseBodyProperties ? 0 : 1;
		stepContext->world->metalLastBodyPropertyUploadBytes = reuseBodyProperties ? 0 : propertyBytes;
		if ( prepareContactsOnGpu &&
			b3MetalPackContactPrepareIndices( context, stepContext, &preparedHasRestitution ) == false )
		{
			return false;
		}
		bool treeRefitEncoded = false;
		bool shapeReadbackEncoded = false;
		if ( finalizeBodies )
		{
			b3MetalPackFinalizeProperties( context->finalizePropertiesBuffer.contents, stepContext->sims, bodyCount,
				stepContext->world );
		}
		int shapeCount = finalizeBodies ? b3MetalPackShapeInputs( context, stepContext ) : 0;
		bool constraintsAlreadyShared = contactBytes == 0 || stepContext->wideConstraints == context->contactConstraintBuffer.contents;
		if ( contactBytes > 0 && constraintsAlreadyShared == false && prepareContactsOnGpu == false )
		{
			memcpy( context->contactConstraintBuffer.contents, stepContext->wideConstraints, contactBytes );
		}
		uint8_t* meshContactBase = context->meshContactBuffer.contents;
		uint8_t* meshManifoldBase = context->meshManifoldBuffer.contents;
		bool coloredMeshContactsShared = coloredMeshContactBytes == 0 ||
			stepContext->contactConstraints == (b3ContactConstraint*)meshContactBase;
		bool overflowMeshContactsShared = overflowMeshContactBytes == 0 ||
			overflow->contactConstraints == (b3ContactConstraint*)( meshContactBase + coloredMeshContactBytes );
		bool coloredMeshManifoldsShared = coloredMeshManifoldBytes == 0 ||
			stepContext->manifoldConstraints == (b3ManifoldConstraint*)meshManifoldBase;
		bool overflowMeshManifoldsShared = overflowMeshManifoldBytes == 0 ||
			overflow->manifoldConstraints == (b3ManifoldConstraint*)( meshManifoldBase + coloredMeshManifoldBytes );
		if ( coloredMeshContactBytes > 0 && coloredMeshContactsShared == false )
		{
			memcpy( meshContactBase, stepContext->contactConstraints, coloredMeshContactBytes );
		}
		if ( overflowMeshContactBytes > 0 && overflowMeshContactsShared == false )
		{
			memcpy( meshContactBase + coloredMeshContactBytes, overflow->contactConstraints, overflowMeshContactBytes );
		}
		if ( coloredMeshManifoldBytes > 0 && coloredMeshManifoldsShared == false )
		{
			memcpy( meshManifoldBase, stepContext->manifoldConstraints, coloredMeshManifoldBytes );
		}
		if ( overflowMeshManifoldBytes > 0 && overflowMeshManifoldsShared == false )
		{
			memcpy( meshManifoldBase + coloredMeshManifoldBytes, overflow->manifoldConstraints,
				overflowMeshManifoldBytes );
		}

		b3MetalDistanceJoint* packedDistanceJoints = context->distanceJointBuffer.contents;
		b3MetalParallelJoint* packedParallelJoints = context->parallelJointBuffer.contents;
		for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
		{
			int count = stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = stepContext->jointPrepareSpans[colorIndex].joints;
			int distanceIndex = distanceOffsets[colorIndex];
			int parallelIndex = parallelOffsets[colorIndex];
			for ( int i = 0; i < count; ++i )
			{
				if ( joints[i].type == b3_distanceJoint )
				{
					b3MetalPackDistanceJoint( packedDistanceJoints + distanceIndex++, joints + i );
				}
				else if ( joints[i].type == b3_parallelJoint )
				{
					b3MetalPackParallelJoint( packedParallelJoints + parallelIndex++, joints + i );
				}
			}
		}
		b3MetalJointOverflow* packedOverflow = context->jointOverflowBuffer.contents;
		int overflowDistanceIndex = coloredDistanceCount;
		int overflowParallelIndex = coloredParallelCount;
		for ( int i = 0; i < stepContext->overflowJointConstraintCount; ++i )
		{
			b3JointSim* joint = overflow->jointSims.data + i;
			if ( joint->type == b3_distanceJoint )
			{
				b3MetalPackDistanceJoint( packedDistanceJoints + overflowDistanceIndex, joint );
				packedOverflow[i] = (b3MetalJointOverflow){ 0, (uint32_t)overflowDistanceIndex++ };
			}
			else
			{
				b3MetalPackParallelJoint( packedParallelJoints + overflowParallelIndex, joint );
				packedOverflow[i] = (b3MetalJointOverflow){ 1, (uint32_t)overflowParallelIndex++ };
			}
		}

		b3MetalFusedParams velocityParams = {
			.bodyCount = (uint32_t)bodyCount,
			.h = stepContext->h,
			.maxLinearSpeed = stepContext->maxLinearVelocity,
			.maxAngularSpeed = B3_MAX_ROTATION * stepContext->inv_dt,
			.gravityX = stepContext->world->gravity.x,
			.gravityY = stepContext->world->gravity.y,
			.gravityZ = stepContext->world->gravity.z,
			.integratePosition = 0,
		};
		b3MetalIntegrateParams positionParams = {
			.bodyCount = (uint32_t)bodyCount,
			.h = stepContext->h,
			.maxLinearSpeed = stepContext->maxLinearVelocity,
			.maxAngularSpeed = B3_MAX_ROTATION * stepContext->inv_dt,
		};

		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		NSUInteger velocityWidth = b3MetalThreadgroupWidth( context->integrateUnconstrainedPipeline );
		NSUInteger positionWidth = b3MetalThreadgroupWidth( context->integratePositionsPipeline );
		NSUInteger prepareWidth = b3MetalThreadgroupWidth( context->prepareContactsPipeline );
		NSUInteger warmWidth = b3MetalThreadgroupWidth( context->warmStartContactsPipeline );
		NSUInteger solveWidth = b3MetalThreadgroupWidth( context->solveContactsPipeline );
		NSUInteger restitutionWidth = b3MetalThreadgroupWidth( context->restitutionContactsPipeline );
		NSUInteger storeImpulseWidth = b3MetalThreadgroupWidth( context->storeContactImpulsesPipeline );
		NSUInteger warmMeshWidth = b3MetalThreadgroupWidth( context->warmStartMeshPipeline );
		NSUInteger solveMeshWidth = b3MetalThreadgroupWidth( context->solveMeshPipeline );
		NSUInteger restitutionMeshWidth = b3MetalThreadgroupWidth( context->restitutionMeshPipeline );
		if ( prepareContactsOnGpu )
		{
			*(uint32_t*)context->contactPrepareStatusBuffer.contents = 0;
			b3MetalContactPrepareParams params = {
				.wideCount = (uint32_t)stepContext->wideContactCount,
				.tableCount = (uint32_t)context->convexManifoldTableCount,
				.warmStartScale = stepContext->enableWarmStarting ? 1.0f : 0.0f,
				.invTau = 1.0f / B3_SPECULATIVE_DISTANCE,
				.contactSoftness = stepContext->contactSoftness,
				.staticSoftness = stepContext->staticSoftness,
				.generation = context->contactPrepareGeneration,
			};
			[encoder setComputePipelineState:context->prepareContactsPipeline];
			[encoder setBuffer:context->contactPrepareIndexBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->convexManifoldTableBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:3];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:4];
			[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:5];
			[encoder setBuffer:context->contactPrepareStatusBuffer offset:0 atIndex:6];
			[encoder setBytes:&params length:sizeof( params ) atIndex:7];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)stepContext->wideContactCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( prepareWidth, 1, 1 )];
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		}
		for ( int subStep = 0; subStep < stepContext->subStepCount; ++subStep )
		{
			[encoder setComputePipelineState:context->integrateUnconstrainedPipeline];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:1];
			[encoder setBytes:&velocityParams length:sizeof( velocityParams ) atIndex:2];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( velocityWidth, 1, 1 )];
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

			if ( stepContext->enableWarmStarting )
			{
				b3MetalJointParams jointParams = { .h = stepContext->h, .invH = stepContext->inv_h };
				b3MetalDispatchJointOverflow( encoder, context->warmStartJointOverflowPipeline, context,
					(uint32_t)stepContext->overflowJointConstraintCount, jointParams );
				b3MetalDispatchOverflowMesh( encoder, context->warmStartOverflowPipeline, context->bodyStateBuffer,
					context->meshContactBuffer, context->meshManifoldBuffer, (uint32_t)stepContext->contactConstraintCount,
					(uint32_t)stepContext->overflowContactConstraintCount, (b3MetalContactParams){ 0 } );
				for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
				{
					bool dispatched = false;
					if ( distanceCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->warmStartDistancePipeline,
							context->bodyStateBuffer, context->distanceJointBuffer, (uint32_t)distanceOffsets[colorIndex],
							(uint32_t)distanceCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					if ( parallelCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->warmStartParallelPipeline,
							context->bodyStateBuffer, context->parallelJointBuffer, (uint32_t)parallelOffsets[colorIndex],
							(uint32_t)parallelCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					int wideOffset = stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
					if ( wideCount > 0 )
					{
						b3MetalContactParams params = { .offset = (uint32_t)wideOffset, .count = (uint32_t)wideCount };
						[encoder setComputePipelineState:context->warmStartContactsPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
						[encoder setBytes:&params length:sizeof( params ) atIndex:2];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideCount, 1, 1 )
							threadsPerThreadgroup:MTLSizeMake( warmWidth, 1, 1 )];
						dispatched = true;
					}
					int meshOffset = stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
					if ( meshCount > 0 )
					{
						b3MetalContactParams params = { .offset = (uint32_t)meshOffset, .count = (uint32_t)meshCount };
						[encoder setComputePipelineState:context->warmStartMeshPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->meshContactBuffer offset:0 atIndex:1];
						[encoder setBuffer:context->meshManifoldBuffer offset:0 atIndex:2];
						[encoder setBytes:&params length:sizeof( params ) atIndex:3];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)meshCount, 1, 1 )
							threadsPerThreadgroup:MTLSizeMake( warmMeshWidth, 1, 1 )];
						dispatched = true;
					}
					if ( dispatched ) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
				}
			}

			for ( int iteration = 0; iteration < velocityIterations; ++iteration )
			{
				b3MetalJointParams jointParams = { .h = stepContext->h, .invH = stepContext->inv_h, .useBias = 1 };
				b3MetalDispatchJointOverflow( encoder, context->solveJointOverflowPipeline, context,
					(uint32_t)stepContext->overflowJointConstraintCount, jointParams );
				b3MetalContactParams overflowParams = { .invH = stepContext->inv_h,
					.contactSpeed = -stepContext->world->contactSpeed, .useBias = 1 };
				b3MetalDispatchOverflowMesh( encoder, context->solveOverflowPipeline, context->bodyStateBuffer,
					context->meshContactBuffer, context->meshManifoldBuffer, (uint32_t)stepContext->contactConstraintCount,
					(uint32_t)stepContext->overflowContactConstraintCount, overflowParams );
				for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
				{
					bool dispatched = false;
					if ( distanceCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->solveDistancePipeline,
							context->bodyStateBuffer, context->distanceJointBuffer, (uint32_t)distanceOffsets[colorIndex],
							(uint32_t)distanceCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					if ( parallelCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->solveParallelPipeline,
							context->bodyStateBuffer, context->parallelJointBuffer, (uint32_t)parallelOffsets[colorIndex],
							(uint32_t)parallelCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					int wideOffset = stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
					b3MetalContactParams common = { .invH = stepContext->inv_h,
						.contactSpeed = -stepContext->world->contactSpeed, .useBias = 1 };
					if ( wideCount > 0 )
					{
						b3MetalContactParams params = common; params.offset = (uint32_t)wideOffset; params.count = (uint32_t)wideCount;
						[encoder setComputePipelineState:context->solveContactsPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
						[encoder setBytes:&params length:sizeof( params ) atIndex:2];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( solveWidth, 1, 1 )];
						dispatched = true;
					}
					int meshOffset = stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
					if ( meshCount > 0 )
					{
						b3MetalContactParams params = common; params.offset = (uint32_t)meshOffset; params.count = (uint32_t)meshCount;
						[encoder setComputePipelineState:context->solveMeshPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->meshContactBuffer offset:0 atIndex:1];
						[encoder setBuffer:context->meshManifoldBuffer offset:0 atIndex:2];
						[encoder setBytes:&params length:sizeof( params ) atIndex:3];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)meshCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( solveMeshWidth, 1, 1 )];
						dispatched = true;
					}
					if ( dispatched ) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
				}
			}

			[encoder setComputePipelineState:context->integratePositionsPipeline];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
			[encoder setBytes:&positionParams length:sizeof( positionParams ) atIndex:1];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( positionWidth, 1, 1 )];
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

			for ( int iteration = 0; iteration < relaxIterations; ++iteration )
			{
				b3MetalJointParams jointParams = { .h = stepContext->h, .invH = stepContext->inv_h, .useBias = 0 };
				b3MetalDispatchJointOverflow( encoder, context->solveJointOverflowPipeline, context,
					(uint32_t)stepContext->overflowJointConstraintCount, jointParams );
				b3MetalContactParams overflowParams = { .invH = stepContext->inv_h,
					.contactSpeed = -stepContext->world->contactSpeed, .useBias = 0 };
				b3MetalDispatchOverflowMesh( encoder, context->solveOverflowPipeline, context->bodyStateBuffer,
					context->meshContactBuffer, context->meshManifoldBuffer, (uint32_t)stepContext->contactConstraintCount,
					(uint32_t)stepContext->overflowContactConstraintCount, overflowParams );
				for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
				{
					bool dispatched = false;
					if ( distanceCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->solveDistancePipeline,
							context->bodyStateBuffer, context->distanceJointBuffer, (uint32_t)distanceOffsets[colorIndex],
							(uint32_t)distanceCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					if ( parallelCounts[colorIndex] > 0 )
					{
						b3MetalDispatchDistanceJoints( encoder, context->solveParallelPipeline,
							context->bodyStateBuffer, context->parallelJointBuffer, (uint32_t)parallelOffsets[colorIndex],
							(uint32_t)parallelCounts[colorIndex], jointParams, false );
						dispatched = true;
					}
					int wideOffset = stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
					b3MetalContactParams common = { .invH = stepContext->inv_h,
						.contactSpeed = -stepContext->world->contactSpeed, .useBias = 0 };
					if ( wideCount > 0 )
					{
						b3MetalContactParams params = common; params.offset = (uint32_t)wideOffset; params.count = (uint32_t)wideCount;
						[encoder setComputePipelineState:context->solveContactsPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
						[encoder setBytes:&params length:sizeof( params ) atIndex:2];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( solveWidth, 1, 1 )];
						dispatched = true;
					}
					int meshOffset = stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
					if ( meshCount > 0 )
					{
						b3MetalContactParams params = common; params.offset = (uint32_t)meshOffset; params.count = (uint32_t)meshCount;
						[encoder setComputePipelineState:context->solveMeshPipeline];
						[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
						[encoder setBuffer:context->meshContactBuffer offset:0 atIndex:1];
						[encoder setBuffer:context->meshManifoldBuffer offset:0 atIndex:2];
						[encoder setBytes:&params length:sizeof( params ) atIndex:3];
						[encoder dispatchThreads:MTLSizeMake( (NSUInteger)meshCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( solveMeshWidth, 1, 1 )];
						dispatched = true;
					}
					if ( dispatched ) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
				}
			}
		}

		bool hasRestitution = prepareContactsOnGpu ? preparedHasRestitution :
			b3MetalHasRestitution( stepContext->wideConstraints, stepContext->wideContactCount );
		bool hasMeshRestitution =
			b3MetalHasMeshRestitution( stepContext->contactConstraints, stepContext->contactConstraintCount );
		bool hasOverflowRestitution =
			b3MetalHasMeshRestitution( overflow->contactConstraints, stepContext->overflowContactConstraintCount );
		for ( int iteration = 0; ( hasRestitution || hasMeshRestitution || hasOverflowRestitution ) && iteration < restitutionIterations;
			  ++iteration )
		{
			if ( hasOverflowRestitution )
			{
				b3MetalContactParams overflowParams = { .restitutionThreshold = stepContext->world->restitutionThreshold };
				b3MetalDispatchOverflowMesh( encoder, context->restitutionOverflowPipeline, context->bodyStateBuffer,
					context->meshContactBuffer, context->meshManifoldBuffer, (uint32_t)stepContext->contactConstraintCount,
					(uint32_t)stepContext->overflowContactConstraintCount, overflowParams );
			}
			for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
			{
				bool dispatched = false;
				int wideOffset = stepContext->widePrepareSpans[colorIndex].start;
				int wideCount = stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
				b3MetalContactParams common = { .restitutionThreshold = stepContext->world->restitutionThreshold };
				if ( hasRestitution && wideCount > 0 )
				{
					b3MetalContactParams params = common; params.offset = (uint32_t)wideOffset; params.count = (uint32_t)wideCount;
					[encoder setComputePipelineState:context->restitutionContactsPipeline];
					[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
					[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
					[encoder setBytes:&params length:sizeof( params ) atIndex:2];
					[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( restitutionWidth, 1, 1 )];
					dispatched = true;
				}
				int meshOffset = stepContext->contactPrepareSpans[colorIndex].start;
				int meshCount = stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
				if ( hasMeshRestitution && meshCount > 0 )
				{
					b3MetalContactParams params = common; params.offset = (uint32_t)meshOffset; params.count = (uint32_t)meshCount;
					[encoder setComputePipelineState:context->restitutionMeshPipeline];
					[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
					[encoder setBuffer:context->meshContactBuffer offset:0 atIndex:1];
					[encoder setBuffer:context->meshManifoldBuffer offset:0 atIndex:2];
					[encoder setBytes:&params length:sizeof( params ) atIndex:3];
					[encoder dispatchThreads:MTLSizeMake( (NSUInteger)meshCount, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( restitutionMeshWidth, 1, 1 )];
					dispatched = true;
				}
				if ( dispatched ) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			}
		}

		if ( prepareContactsOnGpu )
		{
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			b3MetalContactImpulseParams params = {
				.wideCount = (uint32_t)stepContext->wideContactCount,
				.tableCount = (uint32_t)context->convexManifoldTableCount,
				.generation = context->contactPrepareGeneration,
			};
			[encoder setComputePipelineState:context->storeContactImpulsesPipeline];
			[encoder setBuffer:context->contactPrepareIndexBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->contactImpulseResultBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:3];
			[encoder setBytes:&params length:sizeof( params ) atIndex:4];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)stepContext->wideContactCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( storeImpulseWidth, 1, 1 )];
		}

		if ( finalizeBodies )
		{
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			struct { uint32_t bodyCount; float invTimeStep; uint32_t transformCount, publishTransforms; } finalizeParams = {
				(uint32_t)bodyCount, stepContext->inv_dt,
				publishBodyTransforms ? (uint32_t)stepContext->world->bodies.count : 0u,
				publishBodyTransforms ? 1u : 0u,
			};
			id<MTLComputePipelineState> finalizePipeline = context->finalizeBodiesPipeline;
			[encoder setComputePipelineState:finalizePipeline];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->finalizePropertiesBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->finalizeResultBuffer offset:0 atIndex:3];
			[encoder setBytes:&finalizeParams length:sizeof( finalizeParams ) atIndex:4];
			[encoder setBuffer:( publishBodyTransforms ? context->convexBodyTransformBuffer : context->finalizeResultBuffer )
				offset:0 atIndex:5];
			[encoder setBuffer:( publishBodyTransforms ? context->bodyMoveResultBuffer : context->finalizeResultBuffer )
				offset:0 atIndex:6];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( finalizePipeline ), 1, 1 )];
			b3MetalEncodeShapeFinalization( context, encoder, stepContext, shapeCount );
			treeRefitEncoded = b3MetalEncodePairTreeRefit( context, encoder, stepContext, shapeCount );
		}

		[encoder endEncoding];
		if ( finalizeBodies && omitFinalizeReadback == false &&
			b3MetalEncodeFinalizeReadback( context, commandBuffer, bodyCount ) == false )
		{
			return false;
		}
		if ( shapeCount > 0 && ( treeRefitEncoded == false || stepContext->metalDeferShapeResultApply == false ) )
		{
			if ( b3MetalEncodeFullShapeReadback( context, commandBuffer, shapeCount ) == false ) return false;
			shapeReadbackEncoded = true;
		}
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}
		if ( prepareContactsOnGpu && *(const uint32_t*)context->contactPrepareStatusBuffer.contents != 0 )
		{
			return false;
		}
		if ( publishBodyTransforms )
		{
			b3MetalCommitBodyTransformDeviceRefresh( context, stepContext->world, bodyCount );
			context->bodyStateResidentRevision = stepContext->world->metalBodyStateRevision;
			context->bodyPropertiesResidentCount = bodyCount;
			context->bodyPropertiesResidentRevision = stepContext->world->metalBodyPropertyRevision;
			context->bodyMoveResultCount = bodyCount;
			context->bodyMoveResultStepIndex = stepContext->world->stepIndex;
			stepContext->metalBodyStatesFinalizedOnDevice = true;
			stepContext->metalBodyMoveEventsOnDevice = true;
		}
		else
		{
			context->bodyStateResidentCount = 0;
			context->bodyPropertiesResidentCount = 0;
		}
		if ( prepareContactsOnGpu )
		{
			context->contactImpulseResultCount = context->convexManifoldTableCount;
			context->contactImpulseResultGeneration = context->contactPrepareGeneration;
			stepContext->world->metalLastContactImpulseResultBytes =
				(uint64_t)stepContext->world->metalLastResidentConvexContactCount * sizeof( b3MetalContactImpulseResult );
		}

		memcpy( stepContext->states, context->bodyStateBuffer.contents, stateBytes );
		stepContext->world->metalLastBodyStateReadbackBytes = stateBytes;
		if ( finalizeBodies )
		{
			if ( omitFinalizeReadback )
			{
				stepContext->metalFinalizationDeviceOnly = true;
				stepContext->metalFinalizeResults = NULL;
			}
			else
			{
				stepContext->metalFinalizeResults = context->finalizeReadbackBuffer.contents;
			}
		}
		if ( shapeCount > 0 )
		{
			b3MetalAdvanceShapeResultGeneration( stepContext->world );
			context->shapeBoundsCount = shapeCount;
			stepContext->world->metalShapeBoundsResidentDispatchCount += stepContext->metalShapeBoundsResident ? 1 : 0;
			stepContext->metalShapeResults = shapeReadbackEncoded ? context->shapeReadbackBuffer.contents : NULL;
			stepContext->metalShapeResultCount = shapeCount;
			stepContext->world->metalLastShapeResultCount = shapeCount;
			stepContext->world->metalLastEnlargedShapeResultCount = 0;
			if ( treeRefitEncoded )
			{
				b3MetalPairSummary* summary = context->shapeSummaryBuffer.contents;
				if ( summary->totalCount <= (uint64_t)shapeCount )
				{
					stepContext->metalEnlargedShapeResults = context->shapeCompactBuffer.contents;
					stepContext->metalEnlargedShapeResultCount = (int)summary->totalCount;
					stepContext->world->metalLastEnlargedShapeResultCount = (int)summary->totalCount;
					stepContext->world->metalShapeCompactDispatchCount += 1;
				}
				else
				{
					treeRefitEncoded = false;
					if ( shapeReadbackEncoded == false )
					{
						if ( b3MetalReadbackShapeRange( context, 0,
							(NSUInteger)shapeCount * sizeof( b3MetalShapeAABBResult ) ) == false ) return false;
						shapeReadbackEncoded = true;
						stepContext->metalShapeResults = context->shapeReadbackBuffer.contents;
					}
				}
			}
			stepContext->metalTreeRefit = treeRefitEncoded;
			stepContext->world->metalShapeDispatchCount += 1;
			stepContext->world->metalPairTreeRefitCount += treeRefitEncoded ? 1 : 0;
		}
		if ( contactBytes > 0 && constraintsAlreadyShared == false )
		{
			memcpy( stepContext->wideConstraints, context->contactConstraintBuffer.contents, contactBytes );
		}
		if ( coloredMeshContactBytes > 0 && coloredMeshContactsShared == false )
		{
			memcpy( stepContext->contactConstraints, meshContactBase, coloredMeshContactBytes );
		}
		if ( overflowMeshContactBytes > 0 && overflowMeshContactsShared == false )
		{
			memcpy( overflow->contactConstraints, meshContactBase + coloredMeshContactBytes, overflowMeshContactBytes );
		}
		if ( coloredMeshManifoldBytes > 0 && coloredMeshManifoldsShared == false )
		{
			memcpy( stepContext->manifoldConstraints, meshManifoldBase, coloredMeshManifoldBytes );
		}
		if ( overflowMeshManifoldBytes > 0 && overflowMeshManifoldsShared == false )
		{
			memcpy( overflow->manifoldConstraints, meshManifoldBase + coloredMeshManifoldBytes,
				overflowMeshManifoldBytes );
		}
		for ( int colorIndex = 0; colorIndex < stepContext->activeColorCount; ++colorIndex )
		{
			int count = stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = stepContext->jointPrepareSpans[colorIndex].joints;
			int distanceIndex = distanceOffsets[colorIndex];
			int parallelIndex = parallelOffsets[colorIndex];
			for ( int i = 0; i < count; ++i )
			{
				if ( joints[i].type == b3_distanceJoint )
				{
					b3MetalUnpackDistanceJoint( joints + i, packedDistanceJoints + distanceIndex++ );
				}
				else if ( joints[i].type == b3_parallelJoint )
				{
					b3MetalUnpackParallelJoint( joints + i, packedParallelJoints + parallelIndex++ );
				}
			}
		}
		for ( int i = 0; i < stepContext->overflowJointConstraintCount; ++i )
		{
			b3JointSim* joint = overflow->jointSims.data + i;
			b3MetalJointOverflow entry = packedOverflow[i];
			if ( joint->type == b3_distanceJoint )
			{
				b3MetalUnpackDistanceJoint( joint, packedDistanceJoints + entry.index );
			}
			else
			{
				b3MetalUnpackParallelJoint( joint, packedParallelJoints + entry.index );
			}
		}
		if ( stats != NULL && commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
		{
			stats->gpuMilliseconds = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
		}
		return true;
	}
}
