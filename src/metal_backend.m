// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"

#include "constraint_graph.h"
#include "body.h"
#include "broad_phase.h"
#include "contact_solver.h"
#include "hull.h"
#include "joint.h"
#include "physics_world.h"
#include "shape.h"
#include "solver_set.h"

#include "box3d/constants.h"
#include "box3d/types.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dispatch/dispatch.h>


// Phase 0 measurement helpers: wall-clock timing around encode/commit/wait so
// the profile can report CPU encode time and blocking wait time separately
// from GPU execution time. Uses CLOCK_MONOTONIC; never touches Metal state.
static double b3MetalMonotonicMs( void )
{
	struct timespec ts;
	clock_gettime( CLOCK_MONOTONIC, &ts );
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Fill per-dispatch stats from a completed command buffer. Call after
// waitUntilCompleted with encode-side CPU milliseconds measured by the caller.
// When BOX3D_METAL_WAIT_TRACE=1 in the environment, each site also logs one
// line to stderr: site, wait/blocking ms, GPU execution ms, encode ms, and
// dispatch/buffer counts. This is a measure-only spike for Phase 1 async
// planning; the trace is off by default and changes no behavior.
static void b3MetalFillStats( id<MTLCommandBuffer> commandBuffer, b3MetalDispatchStats* stats, double encodeCpuMs,
	double waitCpuMs, int dispatchCount, int barrierCount, int commandBufferCount, const char* site,
	double gpuMsOverride )
{
	double gpuMs = gpuMsOverride >= 0.0 ? gpuMsOverride : 0.0;
	if ( gpuMsOverride < 0.0 && commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
	{
		gpuMs = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
	}
	if ( getenv( "BOX3D_METAL_WAIT_TRACE" ) != NULL )
	{
		fprintf( stderr, "b3metal_wait site=%s wait_ms=%.4f gpu_ms=%.4f encode_ms=%.4f dispatches=%d buffers=%d\n",
			site != NULL ? site : "unknown", waitCpuMs, gpuMs, encodeCpuMs, dispatchCount, commandBufferCount );
	}
	if ( stats == NULL )
	{
		return;
	}
	// Accumulate: pair retry/copy paths submit up to three buffers.
	stats->gpuMilliseconds += gpuMs;
	stats->commandBufferCount += commandBufferCount;
	stats->dispatchCount += dispatchCount;
	stats->barrierCount += barrierCount;
	stats->encodeCpuMilliseconds += encodeCpuMs;
	stats->waitCpuMilliseconds += waitCpuMs;
}

// Phase-1 deferred narrow+solve merge: everything the narrow-phase consume
// step needs after the (possibly shared) command buffer completes. Filled by
// the encode half, read by the consume half; plain C so no ObjC dependency.
typedef struct b3MetalNarrowEncode
{
	bool bootstrapInputs;
	bool pairSeedBootstrapInputs;
	bool privateColdTopology;
	int contactCount;
	int candidateCount;
	uint64_t privateScheduleBytes;
	uint64_t privateScheduleWideCount;
	double encodeMs;
} b3MetalNarrowEncode;

struct b3MetalContext
{
	id<MTLDevice> device;
	id<MTLCommandQueue> queue;
	bool libraryIsBlob;
	bool libraryArchiveHit;
	char libraryCachePath[512];
	// Deferred narrow+solve merge (Phase 1): an encoded-but-uncommitted
	// narrow-phase command buffer plus its consume inputs. The solve phase
	// encodes into the same buffer, commits once, waits once, then consumes
	// the narrow results before its own post step. Nil when nothing pending.
	id<MTLCommandBuffer> pendingNarrowBuffer;
	b3MetalNarrowEncode pendingNarrowEncode;
	bool pendingNarrow;
	// Set by the merged solve wrapper: consume the pending narrow after the
	// single wait. Outcomes reported here for the caller.
	bool mergeConsume;
	bool mergeNarrowOk;
	bool mergeMispredict;
	const b3MetalConvexManifoldResult* mergeResults;
	int mergeResultCount;
	int mergeBypassCount;
	const b3MetalContactTransition* mergeTransitions;
	int mergeTransitionCount;
	b3MetalDispatchStats mergeNarrowStats;
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
	id<MTLComputePipelineState> pairMarkMovesPipeline;
	id<MTLComputePipelineState> pairCompactCpuFilterPipeline;
	id<MTLComputePipelineState> pairContactSeedsPipeline;
	id<MTLComputePipelineState> pairUpdateLeavesPipeline;
	id<MTLComputePipelineState> pairRefitPipeline;
	id<MTLComputePipelineState> contactInputBootstrapPipeline;
	id<MTLComputePipelineState> pairSeedInputBootstrapPipeline;
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
	int finalizePropertiesResidentCount;
	uint64_t finalizePropertiesResidentRevision;
	id<MTLBuffer> bodyMoveResultBuffer;
	NSUInteger bodyMoveResultCapacity;
	id<MTLBuffer> bodyMoveReadbackBuffer;
	NSUInteger bodyMoveReadbackCapacity;
	int bodyMoveResultCount;
	uint64_t bodyMoveResultStepIndex;
	id<MTLBuffer> shapeInputBuffer;
	NSUInteger shapeInputCapacity;
	int shapeInputBodyCount;
	uint64_t shapeInputBodyRevision;
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
	int residentPairMoveCount;
	bool residentPairMovesValid;
	id<MTLBuffer> pairMoveBuffer;
	NSUInteger pairMoveCapacity;
	id<MTLBuffer> pairTreeUploadBuffer;
	NSUInteger pairTreeUploadCapacity;
	id<MTLBuffer> pairTreeBuffer;
	NSUInteger pairTreeCapacity;
	id<MTLBuffer> pairMovedBuffer;
	NSUInteger pairMovedCapacity;
	uint32_t pairMovedEpoch;
	bool pairMovedNeedsClear;
	id<MTLBuffer> pairShapeBuffer;
	NSUInteger pairShapeCapacity;
	uint64_t pairShapeRevision;
	id<MTLBuffer> pairFilterSetBuffer;
	NSUInteger pairFilterSetCapacity;
	uint32_t pairFilterSetItemCapacity;
	uint64_t pairFilterRevision;
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
	id<MTLBuffer> pairPrivateRecordBuffer;
	NSUInteger pairPrivateRecordCapacity;
	id<MTLBuffer> pairPrivateCandidateBuffer;
	NSUInteger pairPrivateCandidateCapacity;
	id<MTLBuffer> pairSummaryBuffer;
	id<MTLBuffer> pairBlockBuffer;
	NSUInteger pairBlockCapacity;
	id<MTLBuffer> pairPrivateBlockBuffer;
	NSUInteger pairPrivateBlockCapacity;
	id<MTLBuffer> pairCpuFilterMoveBuffer;
	NSUInteger pairCpuFilterMoveCapacity;
	id<MTLBuffer> pairContactSeedBuffer;
	NSUInteger pairContactSeedCapacity;
	bool pairShapeMayRequireCpuFiltering;
	bool virginContactInputBootstrapPending;
	int virginContactInputBootstrapCount;
	uint64_t virginContactInputBootstrapPairRevision;
	uint64_t virginContactInputBootstrapGraphRevision;
	uint64_t virginContactInputBootstrapRevision;
	uint64_t virginContactInputBootstrapShapeRevision;
	id<MTLBuffer> convexManifoldInputBuffer;
	NSUInteger convexManifoldInputCapacity;
	id<MTLBuffer> convexManifoldPrivateInputBuffer;
	NSUInteger convexManifoldPrivateInputCapacity;
	bool convexManifoldInputsPrivate;
	int convexManifoldInputCount;
	int convexManifoldCandidateCount;
	uint64_t convexManifoldInputPairRevision;
	uint64_t convexManifoldInputGraphRevision;
	uint64_t convexManifoldInputRevision;
	id<MTLBuffer> contactInputSeedBuffer;
	NSUInteger contactInputSeedCapacity;
	int contactInputSeedBeginCapacity;
	int contactInputBootstrapCount;
	uint64_t contactInputBootstrapPairRevision;
	uint64_t contactInputBootstrapGraphRevision;
	uint64_t contactInputBootstrapRevision;
	bool contactInputBootstrapCommitted;
	bool contactInputBootstrapPairSeeds;
	bool contactInputBootstrapPrivateTopologyCandidate;
	id<MTLBuffer> contactInputBootstrapStatusBuffer;
	id<MTLBuffer> convexManifoldResultBuffer;
	NSUInteger convexManifoldResultCapacity;
	id<MTLBuffer> convexManifoldCompactBuffer;
	NSUInteger convexManifoldCompactCapacity;
	id<MTLBuffer> contactTransitionBuffer;
	NSUInteger contactTransitionCapacity;
	int contactTransitionCount;
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
	id<MTLBuffer> convexHullEdgeBuffer;
	NSUInteger convexHullEdgeCapacity;
	id<MTLBuffer> convexHullFaceBuffer;
	NSUInteger convexHullFaceCapacity;
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
	id<MTLBuffer> privateColdContactScheduleBuffer;
	NSUInteger privateColdContactScheduleCapacity;
	id<MTLBuffer> privateColdBodyOwnerBuffer;
	NSUInteger privateColdBodyOwnerCapacity;
	int privateColdContactScheduleCount;
	int privateColdContactScheduleWideCount;
	uint32_t privateColdContactPrepareGeneration;
	uint64_t privateColdContactPairRevision;
	uint64_t privateColdContactGraphRevision;
	uint64_t privateColdContactInputRevision;
	bool solvingPrivateColdSchedule;
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
	uint32_t cpuFilterMoveCount;
	uint32_t padding;
	uint32_t cpuFilterCandidateCount;
	uint32_t directCandidateCount;
} b3MetalPairSummary;

typedef struct b3MetalPairBlock
{
	uint32_t sum;
	uint32_t flags;
	uint32_t offset;
	uint32_t padding;
} b3MetalPairBlock;

typedef struct b3MetalManifoldBlock
{
	uint32_t exceptionCount;
	uint32_t transitionCount;
	uint32_t stableCount;
	uint32_t silentCount;
	uint32_t exceptionOffset;
	uint32_t transitionOffset;
	uint32_t persistenceMatches;
	uint32_t errorFlags;
} b3MetalManifoldBlock;

typedef struct b3MetalManifoldSummary
{
	uint64_t exceptionCount;
	uint32_t transitionCount;
	uint32_t stableCount;
	uint32_t silentCount;
	uint32_t errorFlags;
	uint64_t persistenceMatches;
} b3MetalManifoldSummary;

typedef struct b3MetalPairShape
{
	int32_t bodyId;
	int32_t sensorIndex;
	int32_t groupIndex;
	uint32_t type;
	uint64_t categoryBits;
	uint64_t maskBits;
} b3MetalPairShape;

// PairShape.type retains its 32-byte ABI. High bits transport host-only shape
// hazards while the low bits remain b3ShapeType.
static const uint32_t b3_metalPairCustomFilterBit = 0x80000000u;
static const uint32_t b3_metalPairContactEventBit = 0x40000000u;
static const uint32_t b3_metalPairHitEventBit = 0x20000000u;
static const uint32_t b3_metalPairPreSolveEventBit = 0x10000000u;

// The b3_finalize_shapes kernel below matches on these literals directly
// (5u = sphere, 0u = capsule, else hull-bounds). Pin them here so any
// b3ShapeType reorder/insertion fails the build instead of silently
// misclassifying geometry on-device.
// Integration kernels test these body-flag literals directly; pin them.

typedef struct b3MetalConvexManifoldInput
{
	uint32_t eligible;
	uint32_t shapeIdA, shapeIdB, contactId;
	uint32_t contactGeneration;
	uint32_t prepareEligible;
	int32_t indexA, indexB;
	float satSeparation;
	uint32_t satCache;
} b3MetalConvexManifoldInput;

typedef struct b3MetalBodyTransform
{
	float qx, qy, qz, qw;
	float px, py, pz;
	uint32_t supported;
	uint64_t pxBits, pyBits, pzBits;
	int32_t index;
	uint32_t flags;
	float localCenterX, localCenterY, localCenterZ, sleepVelocity;
} b3MetalBodyTransform;

typedef struct b3MetalHullTriangle
{
	uint32_t index1, index2, index3, face;
} b3MetalHullTriangle;

typedef struct b3MetalHullEdge
{
	uint32_t origin, twin, next, face;
} b3MetalHullEdge;

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
	uint32_t edgeOffset, edgeCount;
	uint32_t type, supported;
	float friction, restitution, rollingResistance, rollingRadius;
	float tangentVelocityX, tangentVelocityY, tangentVelocityZ, materialPadding;
} b3MetalShapeGeometry;


typedef struct b3MetalFusedParams
{
	uint32_t bodyCount;
	float h;
	float maxLinearSpeed;
	float maxAngularSpeed;
	float gravityX, gravityY, gravityZ;
	uint32_t integratePosition;
	uint32_t subStepCount;
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
	b3MetalContactPreparePoint points[B3_MAX_MANIFOLD_POINTS];
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


typedef struct b3MetalContactImpulseParams
{
	uint32_t wideCount, tableCount, generation, padding;
} b3MetalContactImpulseParams;


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


#include "metal_abi.h"

static void b3MetalWriteError( char* buffer, int capacity, NSString* message )
{
	if ( buffer == NULL || capacity <= 0 )
	{
		return;
	}

	const char* text = message != nil ? message.UTF8String : "unknown Metal error";
	snprintf( buffer, (size_t)capacity, "%s", text );
}

#if defined( BOX3D_METAL_BLOB )
#include "box3d_metallib_blob.h"
#endif
#if defined( BOX3D_METAL_RUNTIME_COMPILE )
#include "metal_sources.h"
#endif

// FNV-1a 64-bit hash for binary-archive cache naming. The hash covers the
// exact bytes the library is built from, so stale archives invalidate when
// shaders change.
static uint64_t b3MetalFnv1a( const void* bytes, size_t count )
{
	const uint8_t* p = (const uint8_t*)bytes;
	uint64_t hash = 14695981039346656037ull;
	for ( size_t i = 0; i < count; ++i )
	{
		hash ^= (uint64_t)p[i];
		hash *= 1099511628211ull;
	}
	return hash;
}

// Best-effort PSO binary-archive cache in ~/Library/Caches/box3d/.
// Writes the cache path into the context for tests; never fails creation.
// Hit means a cache file existed when the archive object was created; the
// archive is (re)populated after PSO creation and serialized for later runs.
// NOTE: MTLComputePipelineDescriptor.binaryArchives consumption is left for a
// later phase; the precompiled-blob path does not need it.
static id<MTLBinaryArchive> b3MetalLoadArchive( id<MTLDevice> device, uint64_t libraryHash, bool* hitOut,
	char* pathOut, int pathCapacity )
{
	if ( hitOut != NULL ) *hitOut = false;
	if ( pathOut != NULL && pathCapacity > 0 ) pathOut[0] = '\0';
	if ( @available( macOS 11.0, * ) )
	{
		char safeName[64];
		const char* deviceName = device.name.UTF8String;
		size_t n = 0;
		for ( size_t i = 0; deviceName != NULL && deviceName[i] != '\0' && n + 1 < sizeof( safeName ); ++i )
		{
			char c = deviceName[i];
			safeName[n++] = ( c >= 'a' && c <= 'z' ) || ( c >= 'A' && c <= 'Z' ) || ( c >= '0' && c <= '9' ) ? c : '_';
		}
		safeName[n] = '\0';
		NSString* dir = [NSString stringWithFormat:@"%@/Library/Caches/box3d", NSHomeDirectory()];
		[[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:NULL];
		NSString* path = [NSString stringWithFormat:@"%@/%s-%016llx.bin", dir, safeName, (unsigned long long)libraryHash];
		if ( pathOut != NULL && pathCapacity > 0 )
		{
			snprintf( pathOut, (size_t)pathCapacity, "%s", path.UTF8String );
		}
		// A descriptor URL opens an existing archive; nil creates an empty
		// one. A non-existent file URL is an error ("Invalid URL").
		bool exists = [[NSFileManager defaultManager] fileExistsAtPath:path];
		MTLBinaryArchiveDescriptor* descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
		descriptor.url = exists ? [NSURL fileURLWithPath:path] : nil;
		NSError* archiveError = nil;
		id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:descriptor error:&archiveError];
		[descriptor release];
		if ( archive != nil && hitOut != NULL )
		{
			*hitOut = exists;
		}
		return archive;
	}
	return nil;
}

// 39 kernels linked into the merged metallib, for archive population.
static const char* b3_metalKernelNames[] = {
	"b3_integrate_positions", "b3_integrate_unconstrained", "b3_finalize_bodies", "b3_finalize_shapes",
	"b3_shape_scan_blocks", "b3_shape_prefix", "b3_shape_scatter", "b3_pair_candidates", "b3_pair_scan_blocks",
	"b3_pair_prefix", "b3_pair_add_offsets", "b3_pair_mark_moves", "b3_pair_compact_cpu_filter",
	"b3_pair_contact_seeds", "b3_pair_update_leaves", "b3_pair_refit", "b3_contact_input_bootstrap",
	"b3_pair_seed_input_bootstrap", "b3_convex_manifolds", "b3_manifold_scan_blocks", "b3_manifold_prefix",
	"b3_manifold_scatter", "b3_prepare_contacts", "b3_store_contact_impulses", "b3_warm_start_contacts",
	"b3_solve_contacts", "b3_restitution_contacts", "b3_warm_start_mesh", "b3_solve_mesh", "b3_restitution_mesh",
	"b3_warm_start_mesh_overflow", "b3_solve_mesh_overflow", "b3_restitution_mesh_overflow", "b3_warm_start_distance",
	"b3_solve_distance", "b3_warm_start_parallel", "b3_solve_parallel", "b3_warm_start_joint_overflow",
	"b3_solve_joint_overflow",
};

static void b3MetalPopulateArchive( id<MTLDevice> device, id<MTLLibrary> library, id<MTLBinaryArchive> archive,
	NSURL* url )
{
	if ( @available( macOS 11.0, * ) )
	{
		if ( device == nil || library == nil || archive == nil || url == nil ) return;
		size_t count = sizeof( b3_metalKernelNames ) / sizeof( b3_metalKernelNames[0] );
		for ( size_t i = 0; i < count; ++i )
		{
			NSString* name = [NSString stringWithUTF8String:b3_metalKernelNames[i]];
			id<MTLFunction> function = [library newFunctionWithName:name];
			if ( function == nil ) continue;
			MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
			descriptor.computeFunction = function;
			[archive addComputePipelineFunctionsWithDescriptor:descriptor error:NULL];
			[descriptor release];
			[function release];
		}
		[archive serializeToURL:url error:NULL];
	}
}

#if defined( BOX3D_METAL_BLOB )
// Load the build-time precompiled metallib embedded as a byte blob. The blob
// lives in the binary's __DATA, so the destructor is a no-op.
static id<MTLLibrary> b3MetalLoadBlobLibrary( id<MTLDevice> device, NSError** errorOut )
{
	if ( b3_metallibBlobLength == 0 ) return nil;
	dispatch_data_t data = dispatch_data_create( b3_metallibBlob, (size_t)b3_metallibBlobLength, NULL, ^(void) {
	} );
	id<MTLLibrary> library = [device newLibraryWithData:data error:errorOut];
	dispatch_release( data );
	return library;
}
#endif

#if defined( BOX3D_METAL_RUNTIME_COMPILE )
// Compile the merged MSL from the generated fallback strings (float and
// double builds each carry their exact historical concatenation).
static id<MTLLibrary> b3MetalCompileSourceLibrary( id<MTLDevice> device, MTLCompileOptions* options,
	NSError** errorOut )
{
	NSString* main = [NSString stringWithUTF8String:b3_metalFallbackMain];
	NSString* contact = [NSString stringWithUTF8String:b3_metalFallbackContact];
	if ( main == nil || contact == nil ) return nil;
	NSMutableString* source = [NSMutableString stringWithString:main];
	[source appendString:contact];
	return [device newLibraryWithSource:source options:options error:errorOut];
}
#endif

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

		// Phase 5a library loading: prefer the build-time precompiled metallib
		// blob (newLibraryWithData), fall back to runtime source compilation
		// when BOX3D_METAL_RUNTIME_COMPILE is on. BOX3D_METAL_FORCE_SOURCE
		// forces the source path for tests.
		bool libraryIsBlob = false;
		bool libraryArchiveHit = false;
		char libraryCachePath[512] = { 0 };
		id<MTLLibrary> library = nil;
		bool forceSource = getenv( "BOX3D_METAL_FORCE_SOURCE" ) != NULL;
#if defined( BOX3D_METAL_BLOB )
		if ( forceSource == false )
		{
			library = b3MetalLoadBlobLibrary( device, &error );
			libraryIsBlob = library != nil;
		}
#endif
		id<MTLBinaryArchive> archive = nil;
		NSURL* archiveURL = nil;
		if ( library != nil )
		{
			// Hash the exact bytes the library was built from so shader edits
			// invalidate stale PSO archives.
#if defined( BOX3D_METAL_BLOB )
			uint64_t libraryHash = b3MetalFnv1a( b3_metallibBlob, (size_t)b3_metallibBlobLength );
#else
			uint64_t libraryHash = 0;
#endif
			archive = b3MetalLoadArchive( device, libraryHash, &libraryArchiveHit, libraryCachePath,
				sizeof( libraryCachePath ) );
			if ( archive != nil && libraryCachePath[0] != '\0' )
			{
				archiveURL = [[NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryCachePath]] retain];
			}
		}
#if defined( BOX3D_METAL_RUNTIME_COMPILE )
		if ( library == nil && forceSource == false )
		{
			// Blob missing or failed: fall through to source compilation.
			error = nil;
		}
		if ( library == nil )
		{
			library = b3MetalCompileSourceLibrary( device, options, &error );
			if ( library != nil && archive == nil )
			{
				// Source path without a cached archive: hash the fallback
				// sources and retry archive acquisition for population.
				size_t mainLen = strlen( b3_metalFallbackMain );
				size_t contactLen = strlen( b3_metalFallbackContact );
				uint64_t libraryHash = b3MetalFnv1a( b3_metalFallbackMain, mainLen );
				libraryHash ^= b3MetalFnv1a( b3_metalFallbackContact, contactLen ) + 0x9e3779b97f4a7c15ull +
					( libraryHash << 6 ) + ( libraryHash >> 2 );
				archive = b3MetalLoadArchive( device, libraryHash, &libraryArchiveHit, libraryCachePath,
					sizeof( libraryCachePath ) );
				if ( archive != nil && libraryCachePath[0] != '\0' )
				{
					archiveURL = [[NSURL fileURLWithPath:[NSString stringWithUTF8String:libraryCachePath]] retain];
				}
			}
		}
#endif
		if ( library == nil )
		{
			[archiveURL release];
			[archive release];
			[options release];
			[queue release];
			[device release];
			b3MetalWriteError( errorBuffer, errorCapacity, error.localizedDescription );
			return false;
		}
		// One merged library serves every kernel; contactLibrary is a retained
		// alias so the per-stage PSO code below is unchanged.
		id<MTLLibrary> contactLibrary = [library retain];

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
		id<MTLFunction> pairMarkMovesFunction = [library newFunctionWithName:@"b3_pair_mark_moves"];
		id<MTLComputePipelineState> pairMarkMovesPipeline = pairMarkMovesFunction != nil
			? [device newComputePipelineStateWithFunction:pairMarkMovesFunction error:&error]
			: nil;
		[pairMarkMovesFunction release];
		id<MTLFunction> pairCompactCpuFilterFunction = [library newFunctionWithName:@"b3_pair_compact_cpu_filter"];
		id<MTLComputePipelineState> pairCompactCpuFilterPipeline = pairCompactCpuFilterFunction != nil
			? [device newComputePipelineStateWithFunction:pairCompactCpuFilterFunction error:&error]
			: nil;
		[pairCompactCpuFilterFunction release];
		id<MTLFunction> pairContactSeedsFunction = [library newFunctionWithName:@"b3_pair_contact_seeds"];
		id<MTLComputePipelineState> pairContactSeedsPipeline = pairContactSeedsFunction != nil
			? [device newComputePipelineStateWithFunction:pairContactSeedsFunction error:&error]
			: nil;
		[pairContactSeedsFunction release];
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
		id<MTLFunction> contactInputBootstrapFunction = [library newFunctionWithName:@"b3_contact_input_bootstrap"];
		id<MTLComputePipelineState> contactInputBootstrapPipeline = contactInputBootstrapFunction != nil
			? [device newComputePipelineStateWithFunction:contactInputBootstrapFunction error:&error]
			: nil;
		[contactInputBootstrapFunction release];
		id<MTLFunction> pairSeedInputBootstrapFunction = [library newFunctionWithName:@"b3_pair_seed_input_bootstrap"];
		id<MTLComputePipelineState> pairSeedInputBootstrapPipeline = pairSeedInputBootstrapFunction != nil
			? [device newComputePipelineStateWithFunction:pairSeedInputBootstrapFunction error:&error]
			: nil;
		[pairSeedInputBootstrapFunction release];
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

		// contactLibrary is the retained alias created with library above.
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
		// All 41 PSOs built: warm the binary-archive cache (best effort) and
		// drop the library references; pipelines stay alive in the context.
		b3MetalPopulateArchive( device, library, archive, archiveURL );
		[archiveURL release];
		[archive release];
		[library release];
		if ( positionPipeline == nil || fusedPipeline == nil || finalizePipeline == nil || finalizeShapesPipeline == nil ||
			 shapeScanBlocksPipeline == nil || shapePrefixPipeline == nil || shapeScatterPipeline == nil ||
			 pairCandidatesPipeline == nil || pairScanBlocksPipeline == nil || pairPrefixPipeline == nil ||
			 pairAddOffsetsPipeline == nil || pairMarkMovesPipeline == nil || pairCompactCpuFilterPipeline == nil ||
			 pairContactSeedsPipeline == nil ||
			 pairUpdateLeavesPipeline == nil || pairRefitPipeline == nil || contactInputBootstrapPipeline == nil ||
			 pairSeedInputBootstrapPipeline == nil ||
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
			[pairMarkMovesPipeline release];
			[pairCompactCpuFilterPipeline release];
			[pairContactSeedsPipeline release];
			[pairUpdateLeavesPipeline release];
			[pairRefitPipeline release];
			[contactInputBootstrapPipeline release];
			[pairSeedInputBootstrapPipeline release];
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
			[pairMarkMovesPipeline release];
			[pairCompactCpuFilterPipeline release];
			[pairContactSeedsPipeline release];
			[pairUpdateLeavesPipeline release];
			[pairRefitPipeline release];
			[contactInputBootstrapPipeline release];
			[pairSeedInputBootstrapPipeline release];
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
		context->libraryIsBlob = libraryIsBlob;
		context->libraryArchiveHit = libraryArchiveHit;
		snprintf( context->libraryCachePath, sizeof( context->libraryCachePath ), "%s", libraryCachePath );
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
		context->pairMarkMovesPipeline = pairMarkMovesPipeline;
		context->pairCompactCpuFilterPipeline = pairCompactCpuFilterPipeline;
		context->pairContactSeedsPipeline = pairContactSeedsPipeline;
		context->pairUpdateLeavesPipeline = pairUpdateLeavesPipeline;
		context->pairRefitPipeline = pairRefitPipeline;
		context->contactInputBootstrapPipeline = contactInputBootstrapPipeline;
		context->pairSeedInputBootstrapPipeline = pairSeedInputBootstrapPipeline;
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

	[context->pendingNarrowBuffer release];
	context->pendingNarrowBuffer = nil;
	[context->bodyStateBuffer release];
	[context->bodyPropertiesBuffer release];
	[context->finalizeResultBuffer release];
	[context->finalizeReadbackBuffer release];
	[context->finalizePropertiesBuffer release];
	[context->bodyMoveResultBuffer release];
	[context->bodyMoveReadbackBuffer release];
	[context->shapeInputBuffer release];
	[context->shapeResultBuffer release];
	[context->shapeReadbackBuffer release];
	[context->shapeCompactBuffer release];
	[context->shapeBlockBuffer release];
	[context->shapeSummaryBuffer release];
	[context->pairMoveBuffer release];
	[context->pairTreeUploadBuffer release];
	[context->pairTreeBuffer release];
	[context->pairMovedBuffer release];
	[context->pairShapeBuffer release];
	[context->pairFilterSetBuffer release];
	[context->pairSetBuffer release];
	[context->pairRecordBuffer release];
	[context->pairCandidateBuffer release];
	[context->pairPrivateRecordBuffer release];
	[context->pairPrivateCandidateBuffer release];
	[context->pairSummaryBuffer release];
	[context->pairBlockBuffer release];
	[context->pairPrivateBlockBuffer release];
	[context->pairCpuFilterMoveBuffer release];
	[context->pairContactSeedBuffer release];
	[context->convexManifoldInputBuffer release];
	[context->convexManifoldPrivateInputBuffer release];
	[context->contactInputSeedBuffer release];
	[context->contactInputBootstrapStatusBuffer release];
	[context->convexManifoldResultBuffer release];
	[context->convexManifoldCompactBuffer release];
	[context->contactTransitionBuffer release];
	[context->privateColdContactScheduleBuffer release];
	[context->privateColdBodyOwnerBuffer release];
	[context->convexManifoldBlockBuffer release];
	[context->convexManifoldSummaryBuffer release];
	[context->convexManifoldTableBuffer release];
	[context->convexManifoldTableReadbackBuffer release];
	[context->convexHullPointBuffer release];
	[context->convexHullPlaneBuffer release];
	[context->convexHullTriangleBuffer release];
	[context->convexHullEdgeBuffer release];
	[context->convexHullFaceBuffer release];
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
	[context->pairMarkMovesPipeline release];
	[context->pairCompactCpuFilterPipeline release];
	[context->pairContactSeedsPipeline release];
	[context->pairUpdateLeavesPipeline release];
	[context->pairRefitPipeline release];
	[context->contactInputBootstrapPipeline release];
	[context->pairSeedInputBootstrapPipeline release];
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

void b3MetalGetLibraryInfo( const b3MetalContext* context, bool* isBlobOut, bool* archiveHitOut, char* pathBuffer,
	int pathCapacity )
{
	if ( isBlobOut != NULL ) *isBlobOut = context != NULL && context->libraryIsBlob;
	if ( archiveHitOut != NULL ) *archiveHitOut = context != NULL && context->libraryArchiveHit;
	if ( pathBuffer != NULL && pathCapacity > 0 )
	{
		snprintf( pathBuffer, (size_t)pathCapacity, "%s", context != NULL ? context->libraryCachePath : "" );
	}
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
	// The buffer identity changed, so any cached residency is invalid.
	// Matches b3MetalEnsurePropertiesCapacity / FinalizeProperties behavior.
	context->bodyStateResidentCount = 0;
	context->bodyStateResidentRevision = 0;
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
	context->finalizePropertiesResidentCount = 0;
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
	NSUInteger compactBytes, NSUInteger transitionBytes, NSUInteger blockBytes, NSUInteger tableBytes, bool privateInput )
{
	if ( context->convexManifoldSummaryBuffer == nil )
	{
		context->convexManifoldSummaryBuffer =
			[context->device newBufferWithLength:sizeof( b3MetalManifoldSummary ) options:MTLResourceStorageModeShared];
		if ( context->convexManifoldSummaryBuffer == nil ) return false;
	}
	if ( privateInput && context->convexManifoldPrivateInputCapacity < inputBytes )
	{
		NSUInteger capacity = context->convexManifoldPrivateInputCapacity > 0 ? context->convexManifoldPrivateInputCapacity : 4096;
		while ( capacity < inputBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->convexManifoldPrivateInputBuffer release];
		context->convexManifoldPrivateInputBuffer = buffer;
		context->convexManifoldPrivateInputCapacity = capacity;
	}
	else if ( privateInput == false && context->convexManifoldInputCapacity < inputBytes )
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
	if ( context->contactTransitionCapacity < transitionBytes )
	{
		NSUInteger capacity = context->contactTransitionCapacity > 0 ? context->contactTransitionCapacity : 4096;
		while ( capacity < transitionBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->contactTransitionBuffer release];
		context->contactTransitionBuffer = buffer;
		context->contactTransitionCapacity = capacity;
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
		context->convexManifoldTableCount = 0;
	}
	return true;
}

static bool b3MetalEnsureShapeGeometryCapacity( b3MetalContext* context, int recordCount, NSUInteger hullPointBytes,
	NSUInteger hullPlaneBytes, NSUInteger hullTriangleBytes, NSUInteger hullEdgeBytes, NSUInteger hullFaceBytes )
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
	hullEdgeBytes = hullEdgeBytes > sizeof( b3MetalHullEdge ) ? hullEdgeBytes : sizeof( b3MetalHullEdge );
	hullFaceBytes = hullFaceBytes > sizeof( uint32_t ) ? hullFaceBytes : sizeof( uint32_t );
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
	if ( context->convexHullEdgeCapacity < hullEdgeBytes )
	{
		NSUInteger capacity = context->convexHullEdgeCapacity > 0 ? context->convexHullEdgeCapacity : 4096;
		while ( capacity < hullEdgeBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexHullEdgeBuffer release];
		context->convexHullEdgeBuffer = buffer;
		context->convexHullEdgeCapacity = capacity;
	}
	if ( context->convexHullFaceCapacity < hullFaceBytes )
	{
		NSUInteger capacity = context->convexHullFaceCapacity > 0 ? context->convexHullFaceCapacity : 4096;
		while ( capacity < hullFaceBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->convexHullFaceBuffer release];
		context->convexHullFaceBuffer = buffer;
		context->convexHullFaceCapacity = capacity;
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
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
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
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->shapeBlockBuffer release];
		context->shapeBlockBuffer = buffer;
		context->shapeBlockCapacity = capacity;
	}
	return true;
}

static bool b3MetalEnsurePairCapacity( b3MetalContext* context, NSUInteger moveBytes, NSUInteger treeBytes,
	NSUInteger movedBytes, NSUInteger shapeBytes, NSUInteger setBytes, NSUInteger recordBytes, NSUInteger candidateBytes,
	NSUInteger blockBytes, NSUInteger contactSeedBytes, bool privateScratch )
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
	if ( privateScratch == false && context->pairCpuFilterMoveCapacity < moveBytes )
	{
		NSUInteger capacity = context->pairCpuFilterMoveCapacity > 0 ? context->pairCpuFilterMoveCapacity : 4096;
		while ( capacity < moveBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairCpuFilterMoveBuffer release];
		context->pairCpuFilterMoveBuffer = buffer;
		context->pairCpuFilterMoveCapacity = capacity;
	}
	if ( context->pairTreeCapacity < treeBytes || context->pairTreeUploadCapacity < treeBytes )
	{
		NSUInteger capacity = context->pairTreeCapacity > 0 ? context->pairTreeCapacity : 4096;
		while ( capacity < treeBytes ) capacity *= 2;
		id<MTLBuffer> uploadBuffer =
			[context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		id<MTLBuffer> treeBuffer =
			[context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( uploadBuffer == nil || treeBuffer == nil )
		{
			[uploadBuffer release];
			[treeBuffer release];
			return false;
		}
		[context->pairTreeUploadBuffer release];
		[context->pairTreeBuffer release];
		context->pairTreeUploadBuffer = uploadBuffer;
		context->pairTreeUploadCapacity = capacity;
		context->pairTreeBuffer = treeBuffer;
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
		context->pairMovedNeedsClear = true;
		context->pairMovedEpoch = 0;
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
	if ( privateScratch && context->pairPrivateRecordCapacity < recordBytes )
	{
		NSUInteger capacity = context->pairPrivateRecordCapacity > 0 ? context->pairPrivateRecordCapacity : 4096;
		while ( capacity < recordBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->pairPrivateRecordBuffer release];
		context->pairPrivateRecordBuffer = buffer;
		context->pairPrivateRecordCapacity = capacity;
	}
	else if ( privateScratch == false && context->pairRecordCapacity < recordBytes )
	{
		NSUInteger capacity = context->pairRecordCapacity > 0 ? context->pairRecordCapacity : 4096;
		while ( capacity < recordBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairRecordBuffer release];
		context->pairRecordBuffer = buffer;
		context->pairRecordCapacity = capacity;
	}
	if ( privateScratch && context->pairPrivateBlockCapacity < blockBytes )
	{
		NSUInteger capacity = context->pairPrivateBlockCapacity > 0 ? context->pairPrivateBlockCapacity : 4096;
		while ( capacity < blockBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->pairPrivateBlockBuffer release];
		context->pairPrivateBlockBuffer = buffer;
		context->pairPrivateBlockCapacity = capacity;
	}
	else if ( privateScratch == false && context->pairBlockCapacity < blockBytes )
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
	if ( privateScratch && context->pairPrivateCandidateCapacity < candidateBytes )
	{
		NSUInteger capacity = context->pairPrivateCandidateCapacity > 0 ? context->pairPrivateCandidateCapacity : 4096;
		while ( capacity < candidateBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->pairPrivateCandidateBuffer release];
		context->pairPrivateCandidateBuffer = buffer;
		context->pairPrivateCandidateCapacity = capacity;
	}
	else if ( privateScratch == false && context->pairCandidateCapacity < candidateBytes )
	{
		NSUInteger capacity = context->pairCandidateCapacity > 0 ? context->pairCandidateCapacity : 4096;
		while ( capacity < candidateBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairCandidateBuffer release];
		context->pairCandidateBuffer = buffer;
		context->pairCandidateCapacity = capacity;
	}
	contactSeedBytes = contactSeedBytes > 0 ? contactSeedBytes : sizeof( b3MetalPairContactSeed );
	if ( context->pairContactSeedCapacity < contactSeedBytes )
	{
		NSUInteger capacity = context->pairContactSeedCapacity > 0 ? context->pairContactSeedCapacity : 4096;
		while ( capacity < contactSeedBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return false;
		[context->pairContactSeedBuffer release];
		context->pairContactSeedBuffer = buffer;
		context->pairContactSeedCapacity = capacity;
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

static bool b3MetalEnsurePrivateColdTopologyCapacity( b3MetalContext* context, NSUInteger scheduleBytes,
	NSUInteger bodyOwnerBytes )
{
	if ( scheduleBytes == 0 || bodyOwnerBytes == 0 ) return false;
	if ( context->privateColdContactScheduleBuffer == nil || context->privateColdContactScheduleCapacity < scheduleBytes )
	{
		NSUInteger capacity = context->privateColdContactScheduleCapacity > 0 ? context->privateColdContactScheduleCapacity : 4096;
		while ( capacity < scheduleBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->privateColdContactScheduleBuffer release];
		context->privateColdContactScheduleBuffer = buffer;
		context->privateColdContactScheduleCapacity = capacity;
	}
	if ( context->privateColdBodyOwnerBuffer == nil || context->privateColdBodyOwnerCapacity < bodyOwnerBytes )
	{
		NSUInteger capacity = context->privateColdBodyOwnerCapacity > 0 ? context->privateColdBodyOwnerCapacity : 4096;
		while ( capacity < bodyOwnerBytes ) capacity *= 2;
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModePrivate];
		if ( buffer == nil ) return false;
		[context->privateColdBodyOwnerBuffer release];
		context->privateColdBodyOwnerBuffer = buffer;
		context->privateColdBodyOwnerCapacity = capacity;
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
		result->pointCount < 1 || result->pointCount > B3_MAX_MANIFOLD_POINTS )
	{
		return false;
	}

	int resultPointIndices[B3_MAX_MANIFOLD_POINTS] = {
		B3_NULL_INDEX, B3_NULL_INDEX, B3_NULL_INDEX, B3_NULL_INDEX,
	};
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

static void b3MetalPackBodyPropertiesRange( b3MetalBodyProperties* properties, const b3BodySim* sims, int begin,
	int end )
{
	for ( int i = begin; i < end; ++i )
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

static void b3MetalPackBodyProperties( b3MetalBodyProperties* properties, const b3BodySim* sims, int bodyCount )
{
	// Streaming pack is memory-bound; spread large packs over all cores in
	// coarse chunks to keep dispatch overhead negligible.
	if ( bodyCount >= 8192 )
	{
		long workerCount = [[NSProcessInfo processInfo] activeProcessorCount];
		if ( workerCount < 2 ) workerCount = 2;
		if ( workerCount > 16 ) workerCount = 16;
		int chunk = ( bodyCount + (int)workerCount - 1 ) / (int)workerCount;
		dispatch_apply( (size_t)workerCount, dispatch_get_global_queue( QOS_CLASS_USER_INITIATED, 0 ), ^( size_t w ) {
			int begin = (int)w * chunk;
			int end = begin + chunk < bodyCount ? begin + chunk : bodyCount;
			if ( begin < end ) b3MetalPackBodyPropertiesRange( properties, sims, begin, end );
		} );
		return;
	}
	b3MetalPackBodyPropertiesRange( properties, sims, 0, bodyCount );
}

bool b3MetalStageResidentContactPrepare( b3MetalContext* context, b3Contact* contact )
{
	if ( context == NULL || contact == NULL || context->contactPrepareGeneration == 0 ||
		contact->contactId < 0 || contact->contactId >= context->convexManifoldTableCount ||
		contact->manifoldCount != 1 || contact->manifolds == NULL ||
		contact->manifolds[0].pointCount < 1 || contact->manifolds[0].pointCount > B3_MAX_MANIFOLD_POINTS )
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
			candidate->contactGeneration == contact->generation && candidate->pointCount >= 1 &&
			candidate->pointCount <= B3_MAX_MANIFOLD_POINTS )
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
	if ( context->solvingPrivateColdSchedule )
	{
		int contactCount = 0, wideCount = 0;
		if ( b3MetalHasPrivateColdContactSchedule( context, stepContext->world, &contactCount, &wideCount ) == false ||
			contactCount != stepContext->metalResidentConvexContactCount || wideCount != stepContext->wideContactCount ||
			context->privateColdContactScheduleCapacity < (NSUInteger)wideCount * B3_SIMD_WIDTH * sizeof( uint32_t ) )
		{
			return false;
		}
		if ( context->contactPrepareStatusBuffer == nil )
		{
			context->contactPrepareStatusBuffer =
				[context->device newBufferWithLength:sizeof( uint32_t ) options:MTLResourceStorageModeShared];
			if ( context->contactPrepareStatusBuffer == nil ) return false;
		}
		stepContext->world->metalLastContactPrepareIndexBytes = 0;
		stepContext->world->metalContactScheduleReuseCount += 1;
		if ( hasRestitution != NULL ) *hasRestitution = true;
		return true;
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

static void b3MetalPackFinalizePropertiesRange( b3MetalFinalizeProperties* properties, const b3BodySim* sims,
	const b3World* world, int begin, int end )
{
	for ( int i = begin; i < end; ++i )
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

static void b3MetalPackFinalizeProperties( b3MetalFinalizeProperties* properties, const b3BodySim* sims, int bodyCount,
	const b3World* world )
{
	if ( bodyCount >= 8192 )
	{
		long workerCount = [[NSProcessInfo processInfo] activeProcessorCount];
		if ( workerCount < 2 ) workerCount = 2;
		if ( workerCount > 16 ) workerCount = 16;
		int chunk = ( bodyCount + (int)workerCount - 1 ) / (int)workerCount;
		dispatch_apply( (size_t)workerCount, dispatch_get_global_queue( QOS_CLASS_USER_INITIATED, 0 ), ^( size_t w ) {
			int begin = (int)w * chunk;
			int end = begin + chunk < bodyCount ? begin + chunk : bodyCount;
			if ( begin < end ) b3MetalPackFinalizePropertiesRange( properties, sims, world, begin, end );
		} );
		return;
	}
	b3MetalPackFinalizePropertiesRange( properties, sims, world, 0, bodyCount );
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
	stepContext->metalShapeFinalizationComplete = false;
	stepContext->metalDeferShapeResultApply = false;
	stepContext->metalTreeRefitEligible = false;
	stepContext->metalTreeRefit = false;
	b3World* world = stepContext->world;
	b3BodySim* sims = stepContext->sims;
	int bodyCount = world->solverSets.data[b3_awakeSet].bodySims.count;
	bool treeRefitEligible = world->metalBroadPhaseEnabled && world->userTreeTask == NULL &&
		context->pairTreeBuffer != nil && context->pairTreeRevision == world->broadPhase.treeRevision;
	bool bodyOrderMatches = context->shapeInputCacheValid && context->shapeInputBodyCount == bodyCount &&
		context->shapeInputBodyRevision == world->metalAwakeBodyRevision;
	world->metalShapeInputOrderRevisionCheckCount += 1;
	if ( world->enableContinuous )
	{
		for ( int simIndex = 0; simIndex < bodyCount; ++simIndex )
		{
			if ( sims[simIndex].flags & b3_isFast )
			{
				treeRefitEligible = false;
				break;
			}
		}
	}

	bool boundsResident = context->shapeBoundsRevision == world->broadPhase.treeRevision &&
		context->shapeBoundsCount == context->shapeInputCount;
	if ( bodyOrderMatches && boundsResident && context->shapeInputCount > 0 )
	{
		stepContext->metalShapeBoundsResident = true;
		stepContext->metalShapeFinalizationComplete = true;
		stepContext->metalDeferShapeResultApply = world->enableContinuous == false && world->sensors.count == 0 &&
			( context->shapeInputAllMasksDisabled ||
			  ( world->metalBroadPhaseEnabled && stepContext->metalFullyResidentConvexContacts ) );
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
		stepContext->metalShapeFinalizationComplete = true;
		return 0;
	}
	if ( b3MetalEnsureShapeCapacity( context, (NSUInteger)shapeCount * sizeof( b3MetalShapeInput ),
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
	context->shapeInputBodyRevision = world->metalAwakeBodyRevision;
	context->shapeInputCount = shapeCount;
	context->shapeInputAllMasksDisabled = allMasksDisabled;
	context->shapeInputCacheValid = true;
	stepContext->metalShapeFinalizationComplete = true;
	stepContext->metalDeferShapeResultApply = world->enableContinuous == false && world->sensors.count == 0 &&
		( allMasksDisabled || ( world->metalBroadPhaseEnabled && stepContext->metalFullyResidentConvexContacts ) );
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
	// This dispatch overwrites the resident compact stream. Do not expose an
	// older generation if encoding or command execution later fails.
	context->residentPairMoveCount = 0;
	context->residentPairMovesValid = false;
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

		double encodeStartMs = b3MetalMonotonicMs();
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
		double preWaitMs = b3MetalMonotonicMs();
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		double postWaitMs = b3MetalMonotonicMs();
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}

		memcpy( states, context->bodyStateBuffer.contents, byteCount );
		b3MetalFillStats( commandBuffer, stats, preWaitMs - encodeStartMs, postWaitMs - preWaitMs, 1, 0, 1,
			"integrate_positions", -1.0 );
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
		bool reuseFinalizeProperties = publishBodyTransforms && context->finalizePropertiesResidentCount == bodyCount &&
			context->finalizePropertiesResidentRevision == finalizationContext->world->metalBodyPropertyRevision;
		// A failed command must not leave the previous generation reusable.
		context->bodyStateResidentCount = 0;
		context->bodyPropertiesResidentCount = 0;
		context->finalizePropertiesResidentCount = 0;
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
		if ( finalizeResults != NULL && reuseFinalizeProperties == false )
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
			.subStepCount = (uint32_t)subStepCount,
		};
		B3_ASSERT( params.integratePosition != 0 || params.subStepCount <= 1 );

		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		double encodeStartMs = b3MetalMonotonicMs();
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
		// Substeps are fused inside the kernel: one dispatch loops over all
		// substeps with body state held in registers. This cuts global-memory
		// traffic ~subStepCount x versus one dispatch per substep.
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)bodyCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( groupWidth, 1, 1 )];
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
		double preWaitMs = b3MetalMonotonicMs();
		[commandBuffer waitUntilCompleted];
		double postWaitMs = b3MetalMonotonicMs();
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
			context->finalizePropertiesResidentCount = bodyCount;
			context->finalizePropertiesResidentRevision = finalizationContext->world->metalBodyPropertyRevision;
			context->bodyMoveResultCount = bodyCount;
			context->bodyMoveResultStepIndex = finalizationContext->world->stepIndex;
			finalizationContext->metalBodyStatesFinalizedOnDevice = true;
			finalizationContext->metalBodyMoveEventsOnDevice = true;
		}
		else
		{
			context->bodyStateResidentCount = 0;
			context->bodyPropertiesResidentCount = 0;
			context->finalizePropertiesResidentCount = 0;
		}

		if ( publishBodyTransforms )
		{
			// Finalization consumes and clears delta state in-place in shared Metal
			// storage. Keep that array authoritative until a CPU boundary asks for it.
			finalizationContext->states = context->bodyStateBuffer.contents;
			b3AtomicStoreInt( &finalizationContext->world->metalBodyStateCpuStale, 1 );
			finalizationContext->world->metalLastBodyStateReadbackBytes = 0;
		}
		else
		{
			memcpy( states, context->bodyStateBuffer.contents, stateByteCount );
			if ( finalizationContext != NULL )
			{
				b3AtomicStoreInt( &finalizationContext->world->metalBodyStateCpuStale, 0 );
				finalizationContext->world->metalLastBodyStateReadbackBytes = stateByteCount;
			}
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
					finalizationContext->metalEnlargedShapeResults = NULL;
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
		if ( stats != NULL )
		{
			// Fused single dispatch covers all substeps; the barrier count is
			// one per substep boundary inside the kernel loop model.
			b3MetalFillStats( commandBuffer, stats, preWaitMs - encodeStartMs, postWaitMs - preWaitMs, 1,
				finalizeResults != NULL ? 1 : 0, 1, "integrate_unconstrained", -1.0 );
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
	if ( context != NULL )
	{
		context->bodyPropertiesResidentCount = 0;
		context->finalizePropertiesResidentCount = 0;
	}
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

		double encodeStartMs = b3MetalMonotonicMs();
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
		double preWaitMs = b3MetalMonotonicMs();
		[commandBuffer waitUntilCompleted];
		double postWaitMs = b3MetalMonotonicMs();
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			return false;
		}

		*results = context->finalizeReadbackBuffer.contents;
		b3MetalFillStats( commandBuffer, stats, preWaitMs - encodeStartMs, postWaitMs - preWaitMs, 1, 0, 1,
			"finalize_bodies", -1.0 );
		return true;
	}
}

static uint64_t b3MetalBodyPairKey( int bodyIdA, int bodyIdB )
{
	uint32_t a = (uint32_t)bodyIdA + 1u;
	uint32_t b = (uint32_t)bodyIdB + 1u;
	uint32_t lo = a < b ? a : b;
	uint32_t hi = a < b ? b : a;
	return ( (uint64_t)lo << 32 ) | hi;
}

static uint32_t b3MetalPairKeyHash( uint64_t key )
{
	uint64_t h = key;
	h ^= h >> 33;
	h *= UINT64_C( 0xff51afd7ed558ccd );
	h ^= h >> 33;
	h *= UINT64_C( 0xc4ceb9fe1a85ec53 );
	h ^= h >> 33;
	return (uint32_t)h;
}

static bool b3MetalRefreshPairFilterSet( b3MetalContext* context, const b3World* world, b3MetalDispatchStats* stats )
{
	if ( context->pairFilterSetBuffer != nil && context->pairFilterRevision == world->metalPairFilterRevision ) return true;
	int blockedJointCount = 0;
	for ( int jointId = 0; jointId < world->joints.count; ++jointId )
	{
		const b3Joint* joint = world->joints.data + jointId;
		if ( joint->jointId == B3_NULL_INDEX || joint->collideConnected ) continue;
		int bodyIdA = joint->edges[0].bodyId;
		int bodyIdB = joint->edges[1].bodyId;
		uint64_t key = b3MetalBodyPairKey( bodyIdA, bodyIdB );
		if ( bodyIdA < 0 || bodyIdB < 0 || bodyIdA == bodyIdB || bodyIdA >= world->bodies.count ||
			 bodyIdB >= world->bodies.count || key == 0 || b3MetalPairKeyHash( key ) == 0 ||
			 blockedJointCount >= ( 1 << 29 ) ) return false;
		blockedJointCount += 1;
	}

	b3HashSet blockedPairs = b3CreateSet( 2 * blockedJointCount );
	for ( int jointId = 0; jointId < world->joints.count; ++jointId )
	{
		const b3Joint* joint = world->joints.data + jointId;
		if ( joint->jointId == B3_NULL_INDEX || joint->collideConnected ) continue;
		uint64_t key = b3MetalBodyPairKey( joint->edges[0].bodyId, joint->edges[1].bodyId );
		if ( key == 0 )
		{
			b3DestroySet( &blockedPairs );
			return false;
		}
		b3AddKey( &blockedPairs, key );
	}

	if ( blockedPairs.capacity == 0 || ( blockedPairs.capacity & ( blockedPairs.capacity - 1 ) ) != 0 ||
		 (NSUInteger)blockedPairs.capacity > NSUIntegerMax / sizeof( b3SetItem ) )
	{
		b3DestroySet( &blockedPairs );
		return false;
	}
	NSUInteger bytes = (NSUInteger)blockedPairs.capacity * sizeof( b3SetItem );
	if ( context->pairFilterSetCapacity < bytes )
	{
		NSUInteger capacity = context->pairFilterSetCapacity > 0 ? context->pairFilterSetCapacity : 4096;
		while ( capacity < bytes )
		{
			if ( capacity > NSUIntegerMax / 2 )
			{
				b3DestroySet( &blockedPairs );
				return false;
			}
			capacity *= 2;
		}
		id<MTLBuffer> buffer = [context->device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
		if ( buffer == nil )
		{
			b3DestroySet( &blockedPairs );
			return false;
		}
		[context->pairFilterSetBuffer release];
		context->pairFilterSetBuffer = buffer;
		context->pairFilterSetCapacity = capacity;
	}
	memcpy( context->pairFilterSetBuffer.contents, blockedPairs.items, bytes );
	context->pairFilterSetItemCapacity = blockedPairs.capacity;
	context->pairFilterRevision = world->metalPairFilterRevision;
	b3DestroySet( &blockedPairs );
	if ( stats != NULL ) stats->pairFilterRegistryUploadCount = 1;
	return true;
}

bool b3MetalGeneratePairCandidates( b3MetalContext* context, const b3World* world, const int* moveArray, int moveCount,
	const b3MetalPairQueryRecord** recordsOut,
	const b3MetalPairCandidate** candidatesOut, int* candidateCountOut, const int** cpuFilterMovesOut,
	int* cpuFilterMoveCountOut, const b3MetalPairContactSeed** contactSeedsOut, int* contactSeedCountOut,
	b3MetalDispatchStats* stats )
{
	if ( recordsOut != NULL ) *recordsOut = NULL;
	if ( candidatesOut != NULL ) *candidatesOut = NULL;
	if ( candidateCountOut != NULL ) *candidateCountOut = 0;
	if ( cpuFilterMovesOut != NULL ) *cpuFilterMovesOut = NULL;
	if ( cpuFilterMoveCountOut != NULL ) *cpuFilterMoveCountOut = 0;
	if ( contactSeedsOut != NULL ) *contactSeedsOut = NULL;
	if ( contactSeedCountOut != NULL ) *contactSeedCountOut = 0;
	if ( stats != NULL ) *stats = (b3MetalDispatchStats){ .bodyCount = moveCount };
	if ( context == NULL || world == NULL || moveCount < 0 || recordsOut == NULL ||
		 candidatesOut == NULL || candidateCountOut == NULL || cpuFilterMovesOut == NULL || cpuFilterMoveCountOut == NULL ||
		 contactSeedsOut == NULL || contactSeedCountOut == NULL )
	{
		return false;
	}
	if ( context->contactInputBootstrapPairSeeds ) b3MetalCancelContactInputBootstrap( context );
	b3MetalCancelPrivateColdContactSchedule( context );
	b3MetalCancelVirginContactInputBootstrap( context );
	bool useResidentMoves = moveArray == NULL;
	if ( useResidentMoves && ( context->residentPairMovesValid == false ||
		 context->residentPairMoveCount != moveCount ) ) return false;
	if ( moveCount == 0 )
	{
		if ( useResidentMoves )
		{
			context->residentPairMoveCount = 0;
			context->residentPairMovesValid = false;
		}
		return true;
	}
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
		bool pairShapeMayRequireCpuFiltering = context->pairShapeMayRequireCpuFiltering;
		if ( context->pairShapeBuffer == nil || context->pairShapeRevision != world->metalPairShapeRevision )
		{
			pairShapeMayRequireCpuFiltering = false;
			for ( int shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex )
			{
				const b3Shape* shape = world->shapes.data + shapeIndex;
				if ( shape->id != B3_NULL_INDEX &&
					( shape->type == b3_compoundShape || ( shape->flags & b3_enableCustomFiltering ) != 0 ) )
				{
					pairShapeMayRequireCpuFiltering = true;
					break;
				}
			}
		}
		bool privatePairScratch = pairShapeMayRequireCpuFiltering == false;
		bool graphContactsEmpty = true;
		for ( int colorIndex = 0; colorIndex < B3_GRAPH_COLOR_COUNT; ++colorIndex )
		{
			const b3GraphColor* color = world->constraintGraph.colors + colorIndex;
			if ( color->convexContacts.count != 0 || color->contacts.count != 0 )
			{
				graphContactsEmpty = false;
				break;
			}
		}
		const b3SolverSet* awakeSet = world->solverSets.data + b3_awakeSet;
		bool virginInputPlan = world->contacts.count == 0 && b3GetIdCount( &world->contactIdPool ) == 0 &&
			broadPhase->pairSet.count == 0 && awakeSet->contactIndices.count == 0 && graphContactsEmpty &&
			world->metalDefaultFrictionCallback && world->metalDefaultRestitutionCallback &&
			world->contactRecycleDistance == 0.0f && world->recording == NULL && privatePairScratch;
		uint32_t pairSetCapacity = broadPhase->pairSet.capacity;
		if ( pairSetCapacity == 0 || ( pairSetCapacity & ( pairSetCapacity - 1 ) ) != 0 ||
			 (NSUInteger)pairSetCapacity > NSUIntegerMax / sizeof( b3SetItem ) ) return false;
		NSUInteger setBytes = (NSUInteger)pairSetCapacity * sizeof( b3SetItem );
		uint64_t candidateLimit64 = 64ull * (uint64_t)moveCount;
		uint32_t candidateLimit = candidateLimit64 < INT32_MAX ? (uint32_t)candidateLimit64 : INT32_MAX;
		uint64_t compactCandidateLimit64 = 16ull * (uint64_t)moveCount;
		if ( compactCandidateLimit64 > NSUIntegerMax / sizeof( b3MetalPairContactSeed ) ) return false;
		NSUInteger contactSeedBytes = (NSUInteger)compactCandidateLimit64 * sizeof( b3MetalPairContactSeed );
		uint64_t initialCandidateCount = 4ull * (uint64_t)moveCount;
		if ( initialCandidateCount > candidateLimit ) initialCandidateCount = candidateLimit;
		if ( initialCandidateCount > NSUIntegerMax / sizeof( b3MetalPairCandidate ) ) return false;
		NSUInteger initialCandidateBytes = (NSUInteger)initialCandidateCount * sizeof( b3MetalPairCandidate );
		if ( b3MetalEnsurePairCapacity( context, moveBytes, treeBytes, movedBytes, shapeBytes, setBytes, recordBytes,
			 initialCandidateBytes, blockBytes, contactSeedBytes, privatePairScratch ) == false )
		{
			return false;
		}
		if ( b3MetalRefreshPairFilterSet( context, world, stats ) == false ) return false;
		for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
		{
			const b3DynamicTree* tree = broadPhase->trees + treeIndex;
			context->pairTreeOffsets[treeIndex] = nodeOffsets[treeIndex];
			context->pairTreeNodeCounts[treeIndex] = (uint32_t)tree->nodeCapacity;
			context->pairTreeHeights[treeIndex] = tree->root == B3_NULL_INDEX ? 0 : tree->nodes[tree->root].height;
		}

		if ( useResidentMoves == false ) memcpy( context->pairMoveBuffer.contents, moveArray, moveBytes );
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
				uint32_t pairType = (uint32_t)shape->type;
				pairType |= ( shape->flags & b3_enableCustomFiltering ) != 0 ? b3_metalPairCustomFilterBit : 0u;
				pairType |= ( shape->flags & b3_enableContactEvents ) != 0 ? b3_metalPairContactEventBit : 0u;
				pairType |= ( shape->flags & b3_enableHitEvents ) != 0 ? b3_metalPairHitEventBit : 0u;
				pairType |= ( shape->flags & b3_enablePreSolveEvents ) != 0 ? b3_metalPairPreSolveEventBit : 0u;
				pairShapes[shapeIndex] = (b3MetalPairShape){
					.bodyId = shape->bodyId,
					.sensorIndex = shape->sensorIndex,
					.groupIndex = shape->filter.groupIndex,
					.type = pairType,
					.categoryBits = shape->filter.categoryBits,
					.maskBits = shape->filter.maskBits,
				};
			}
			context->pairShapeMayRequireCpuFiltering = pairShapeMayRequireCpuFiltering;
			context->pairShapeRevision = world->metalPairShapeRevision;
			if ( stats != NULL ) stats->metadataUploadCount = 1;
		}
		if ( context->pairSetRevision != broadPhase->pairSetRevision )
		{
			memcpy( context->pairSetBuffer.contents, broadPhase->pairSet.items, setBytes );
			context->pairSetRevision = broadPhase->pairSetRevision;
			if ( stats != NULL ) stats->pairSetUploadCount = 1;
		}
		if ( useResidentMoves == false )
		{
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
			}
		}
		bool uploadPairTree = context->pairTreeRevision != broadPhase->treeRevision;
		if ( uploadPairTree )
		{
			b3TreeNode* nodeDestination = context->pairTreeUploadBuffer.contents;
			for ( int treeIndex = 0; treeIndex < b3_bodyTypeCount; ++treeIndex )
			{
				const b3DynamicTree* tree = broadPhase->trees + treeIndex;
				memcpy( nodeDestination + nodeOffsets[treeIndex], tree->nodes,
					(NSUInteger)tree->nodeCapacity * sizeof( b3TreeNode ) );
			}
		}
		memset( context->pairSummaryBuffer.contents, 0, sizeof( b3MetalPairSummary ) );
		uint32_t movedEpoch = context->pairMovedEpoch + 1u;
		if ( context->pairMovedNeedsClear || movedEpoch == 0u )
		{
			memset( context->pairMovedBuffer.contents, 0, context->pairMovedCapacity );
			context->pairMovedNeedsClear = false;
			movedEpoch = 1u;
		}
		context->pairMovedEpoch = movedEpoch;

		struct
		{
			int root0, root1, root2;
			uint32_t offset0, offset1, offset2, moveCount, writeCandidates, shapeCount, pairCapacity;
			uint32_t movedEpoch, residentMoves, filterCapacity, padding0, padding1, padding2;
		} params = {
			broadPhase->trees[0].root, broadPhase->trees[1].root, broadPhase->trees[2].root,
			nodeOffsets[0], nodeOffsets[1], nodeOffsets[2], (uint32_t)moveCount, 0, (uint32_t)shapeCount, pairSetCapacity,
			movedEpoch, useResidentMoves ? 1u : 0u, context->pairFilterSetItemCapacity, 0, 0, 0,
		};
		_Static_assert( sizeof( params ) == 64, "Metal pair parameter ABI changed" );
		NSUInteger pairCandidateCapacity =
			privatePairScratch ? context->pairPrivateCandidateCapacity : context->pairCandidateCapacity;
		uint64_t availableCandidateCount64 = pairCandidateCapacity / sizeof( b3MetalPairCandidate );
		uint32_t availableCandidateCount = availableCandidateCount64 < UINT32_MAX
			? (uint32_t)availableCandidateCount64 : UINT32_MAX;
		struct
		{
			uint32_t moveCount, candidateCapacity, candidateLimit, padding;
		} prefixParams = { (uint32_t)moveCount, availableCandidateCount, candidateLimit, 0 };
		_Static_assert( sizeof( prefixParams ) == 16, "Metal pair-prefix parameter ABI changed" );
		id<MTLBuffer> pairRecordBuffer = privatePairScratch ? context->pairPrivateRecordBuffer : context->pairRecordBuffer;
		id<MTLBuffer> pairCandidateBuffer =
			privatePairScratch ? context->pairPrivateCandidateBuffer : context->pairCandidateBuffer;
		id<MTLBuffer> pairBlockBuffer = privatePairScratch ? context->pairPrivateBlockBuffer : context->pairBlockBuffer;

		id<MTLComputePipelineState> pairPipeline = context->pairCandidatesPipeline;
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		if ( commandBuffer == nil ) return false;
		double encodeStartMs = b3MetalMonotonicMs();
		double preWaitMs = 0.0, postWaitMs = 0.0;
		if ( uploadPairTree )
		{
			id<MTLBlitCommandEncoder> uploadEncoder = [commandBuffer blitCommandEncoder];
			if ( uploadEncoder == nil ) return false;
			[uploadEncoder copyFromBuffer:context->pairTreeUploadBuffer sourceOffset:0
				toBuffer:context->pairTreeBuffer destinationOffset:0 size:treeBytes];
			[uploadEncoder endEncoding];
		}
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( encoder == nil ) return false;
		id<MTLBuffer> residentMoveBuffer = context->shapeCompactBuffer != nil ? context->shapeCompactBuffer : context->pairMoveBuffer;
		[encoder setComputePipelineState:context->pairMarkMovesPipeline];
		[encoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairMovedBuffer offset:0 atIndex:1];
		[encoder setBuffer:residentMoveBuffer offset:0 atIndex:2];
		[encoder setBytes:&params length:sizeof( params ) atIndex:3];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairMarkMovesPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:pairPipeline];
		[encoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
		[encoder setBuffer:pairRecordBuffer offset:0 atIndex:2];
		[encoder setBuffer:pairCandidateBuffer offset:0 atIndex:3];
		[encoder setBytes:&params length:sizeof( params ) atIndex:4];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
		[encoder setBuffer:residentMoveBuffer offset:0 atIndex:9];
		[encoder setBuffer:context->pairFilterSetBuffer offset:0 atIndex:10];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairScanBlocksPipeline];
		[encoder setBuffer:pairRecordBuffer offset:0 atIndex:0];
		[encoder setBuffer:pairBlockBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreadgroups:MTLSizeMake( pairBlockCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( 256, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairPrefixPipeline];
		[encoder setBuffer:pairBlockBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreads:MTLSizeMake( 1, 1, 1 ) threadsPerThreadgroup:MTLSizeMake( 1, 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairAddOffsetsPipeline];
		[encoder setBuffer:pairRecordBuffer offset:0 atIndex:0];
		[encoder setBuffer:pairBlockBuffer offset:0 atIndex:1];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairAddOffsetsPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		params.writeCandidates = 1;
		[encoder setComputePipelineState:pairPipeline];
		[encoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
		[encoder setBuffer:pairRecordBuffer offset:0 atIndex:2];
		[encoder setBuffer:pairCandidateBuffer offset:0 atIndex:3];
		[encoder setBytes:&params length:sizeof( params ) atIndex:4];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
		[encoder setBuffer:residentMoveBuffer offset:0 atIndex:9];
		[encoder setBuffer:context->pairFilterSetBuffer offset:0 atIndex:10];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		[encoder setComputePipelineState:context->pairContactSeedsPipeline];
		[encoder setBuffer:pairRecordBuffer offset:0 atIndex:0];
		[encoder setBuffer:pairCandidateBuffer offset:0 atIndex:1];
		[encoder setBuffer:context->pairContactSeedBuffer offset:0 atIndex:2];
		[encoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:3];
		[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:4];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairContactSeedsPipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		if ( privatePairScratch == false )
		{
			[encoder setComputePipelineState:context->pairCompactCpuFilterPipeline];
			[encoder setBuffer:pairRecordBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->pairCpuFilterMoveBuffer offset:0 atIndex:1];
			[encoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:2];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairCompactCpuFilterPipeline ), 1, 1 )];
		}
		[encoder endEncoding];
		preWaitMs = b3MetalMonotonicMs();
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		postWaitMs = b3MetalMonotonicMs();
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted ) return false;
		if ( uploadPairTree ) context->pairTreeRevision = broadPhase->treeRevision;
		if ( stats != NULL )
		{
			// Steady pair schedule: mark-moves, count, scan-blocks, prefix,
			// add-offsets, write, contact-seeds, cpu-filter-compact.
			b3MetalFillStats( commandBuffer, stats, preWaitMs - encodeStartMs, postWaitMs - preWaitMs, 8, 6, 1,
				"pairs", -1.0 );
		}

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
					 blockBytes, contactSeedBytes, privatePairScratch ) == false ) return false;
			pairCandidateBuffer = privatePairScratch ? context->pairPrivateCandidateBuffer : context->pairCandidateBuffer;
			summary = context->pairSummaryBuffer.contents;
			summary->flags = 0;
			summary->writeFlags = 0;
			id<MTLCommandBuffer> retryCommand = [context->queue commandBuffer];
			id<MTLComputeCommandEncoder> retryEncoder = [retryCommand computeCommandEncoder];
			if ( retryCommand == nil || retryEncoder == nil ) return false;
			[retryEncoder setComputePipelineState:pairPipeline];
			[retryEncoder setBuffer:context->pairMoveBuffer offset:0 atIndex:0];
			[retryEncoder setBuffer:context->pairTreeBuffer offset:0 atIndex:1];
			[retryEncoder setBuffer:pairRecordBuffer offset:0 atIndex:2];
			[retryEncoder setBuffer:pairCandidateBuffer offset:0 atIndex:3];
			[retryEncoder setBytes:&params length:sizeof( params ) atIndex:4];
			[retryEncoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:5];
			[retryEncoder setBuffer:context->pairMovedBuffer offset:0 atIndex:6];
			[retryEncoder setBuffer:context->pairShapeBuffer offset:0 atIndex:7];
			[retryEncoder setBuffer:context->pairSetBuffer offset:0 atIndex:8];
			[retryEncoder setBuffer:residentMoveBuffer offset:0 atIndex:9];
			[retryEncoder setBuffer:context->pairFilterSetBuffer offset:0 atIndex:10];
			[retryEncoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pairPipeline ), 1, 1 )];
			[retryEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
			[retryEncoder setComputePipelineState:context->pairContactSeedsPipeline];
			[retryEncoder setBuffer:pairRecordBuffer offset:0 atIndex:0];
			[retryEncoder setBuffer:pairCandidateBuffer offset:0 atIndex:1];
			[retryEncoder setBuffer:context->pairContactSeedBuffer offset:0 atIndex:2];
			[retryEncoder setBuffer:context->pairSummaryBuffer offset:0 atIndex:3];
			[retryEncoder setBytes:&prefixParams length:sizeof( prefixParams ) atIndex:4];
			[retryEncoder dispatchThreads:MTLSizeMake( (NSUInteger)moveCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( context->pairContactSeedsPipeline ), 1, 1 )];
			[retryEncoder endEncoding];
			double retryPreWaitMs = b3MetalMonotonicMs();
			[retryCommand commit];
			[retryCommand waitUntilCompleted];
			double retryPostWaitMs = b3MetalMonotonicMs();
			if ( retryCommand.status != MTLCommandBufferStatusCompleted ) return false;
			if ( stats != NULL )
			{
				b3MetalFillStats( retryCommand, stats, 0.0, retryPostWaitMs - retryPreWaitMs, 2, 1, 1, "pairs_retry", -1.0 );
			}
		}
		if ( summary->flags != 0 || summary->writeFlags != 0 || summary->totalCount > INT32_MAX ||
			 summary->cpuFilterMoveCount > (uint32_t)moveCount ||
			 (uint64_t)summary->cpuFilterCandidateCount + (uint64_t)summary->directCandidateCount != summary->totalCount ) return false;
		if ( privatePairScratch && ( summary->cpuFilterMoveCount != 0 || summary->cpuFilterCandidateCount != 0 ) ) return false;
		bool compactSeedPlan = summary->cpuFilterMoveCount == 0 && summary->totalCount <= compactCandidateLimit64;
		bool virginInputPending = virginInputPlan && compactSeedPlan && summary->totalCount > 0;
		bool materializedPrivateScratch = false;
		if ( privatePairScratch && compactSeedPlan == false )
		{
			// Preserve the legacy <=64 candidates-per-move capability for unusually
			// dense direct plans. This is an explicit escape hatch: the common <=16
			// route never allocates or exposes raw shared traversal storage.
			if ( summary->totalCount > NSUIntegerMax / sizeof( b3MetalPairCandidate ) ) return false;
			NSUInteger candidateBytes = (NSUInteger)summary->totalCount * sizeof( b3MetalPairCandidate );
			if ( b3MetalEnsurePairCapacity( context, moveBytes, treeBytes, movedBytes, shapeBytes, setBytes, recordBytes,
					 candidateBytes, blockBytes, contactSeedBytes, false ) == false ) return false;
			id<MTLCommandBuffer> copyCommand = [context->queue commandBuffer];
			id<MTLBlitCommandEncoder> copyEncoder = [copyCommand blitCommandEncoder];
			if ( copyCommand == nil || copyEncoder == nil ) return false;
			[copyEncoder copyFromBuffer:pairRecordBuffer sourceOffset:0 toBuffer:context->pairRecordBuffer
				destinationOffset:0 size:recordBytes];
			if ( candidateBytes > 0 )
			{
				[copyEncoder copyFromBuffer:pairCandidateBuffer sourceOffset:0 toBuffer:context->pairCandidateBuffer
					destinationOffset:0 size:candidateBytes];
			}
			[copyEncoder endEncoding];
			double copyPreWaitMs = b3MetalMonotonicMs();
			[copyCommand commit];
			[copyCommand waitUntilCompleted];
			double copyPostWaitMs = b3MetalMonotonicMs();
			if ( copyCommand.status != MTLCommandBufferStatusCompleted ) return false;
			materializedPrivateScratch = true;
			if ( stats != NULL )
			{
				b3MetalFillStats( copyCommand, stats, 0.0, copyPostWaitMs - copyPreWaitMs, 0, 0, 1, "pairs_copy", -1.0 );
			}
		}
		if ( virginInputPending )
		{
			context->virginContactInputBootstrapPending = true;
			context->virginContactInputBootstrapCount = (int)summary->totalCount;
			context->virginContactInputBootstrapPairRevision = broadPhase->pairSetRevision;
			context->virginContactInputBootstrapGraphRevision = world->constraintGraph.revision;
			context->virginContactInputBootstrapRevision = world->metalContactInputRevision;
			context->virginContactInputBootstrapShapeRevision = world->metalPairShapeRevision;
		}

		*recordsOut = privatePairScratch && materializedPrivateScratch == false ? NULL : context->pairRecordBuffer.contents;
		*candidatesOut = privatePairScratch && materializedPrivateScratch == false ? NULL : context->pairCandidateBuffer.contents;
		*candidateCountOut = (int)summary->totalCount;
		*cpuFilterMovesOut = privatePairScratch ? NULL : context->pairCpuFilterMoveBuffer.contents;
		*cpuFilterMoveCountOut = (int)summary->cpuFilterMoveCount;
		if ( compactSeedPlan )
		{
			*contactSeedsOut = context->pairContactSeedBuffer.contents;
			*contactSeedCountOut = (int)summary->totalCount;
		}
		if ( useResidentMoves )
		{
			context->residentPairMoveCount = 0;
			context->residentPairMovesValid = false;
		}
		if ( stats != NULL )
		{
			stats->treeUploadCount = uploadPairTree ? 1 : 0;
			stats->pairTreeUploadBytes = uploadPairTree ? treeBytes : 0;
			stats->pairTreePrivateBytes = treeBytes;
			stats->pairRequiresCpuFiltering = summary->cpuFilterMoveCount != 0;
			stats->pairContactSeedCount = compactSeedPlan ? (int)summary->totalCount : 0;
			stats->pairContactSeedDispatchCount = compactSeedPlan && summary->totalCount > 0 ? 1 : 0;
			stats->pairContactSeedSharedBytes = compactSeedPlan
				? summary->totalCount * sizeof( b3MetalPairContactSeed ) : 0;
			stats->pairPrivateScratchDispatchCount = privatePairScratch ? 1 : 0;
			stats->pairRawSharedBytes = privatePairScratch && materializedPrivateScratch == false
				? 0 : recordBytes + summary->totalCount * sizeof( b3MetalPairCandidate );
			stats->pairCpuFilterCandidateCount = (int)summary->cpuFilterCandidateCount;
			stats->pairDirectCandidateCount = (int)summary->directCandidateCount;
		}
		return true;
	}
}

static bool b3MetalHasValidHullTopology( const b3Shape* shape )
{
	if ( shape->type != b3_hullShape || shape->hull == NULL ) return false;
	const b3HullData* hull = shape->hull;
	if ( hull->version != B3_HULL_VERSION || hull->vertexCount < 4 || hull->faceCount < 4 || hull->edgeCount < 12 ||
		 b3GetHullPoints( hull ) == NULL || b3GetHullPlanes( hull ) == NULL || b3GetHullFaces( hull ) == NULL ||
		 b3GetHullEdges( hull ) == NULL )
	{
		return false;
	}
	return true;
}

static bool b3MetalSupportsHull( const b3Shape* shape )
{
	if ( b3MetalHasValidHullTopology( shape ) == false ) return false;
	const b3HullData* hull = shape->hull;
	b3Vec3 extent = b3Sub( hull->aabb.upperBound, hull->aabb.lowerBound );
	float minExtent = b3MinFloat( extent.x, b3MinFloat( extent.y, extent.z ) );
	float maxExtent = b3MaxFloat( extent.x, b3MaxFloat( extent.y, extent.z ) );
	// Boundary-triangle closest points are stable for ordinary compact hulls.
	// High-aspect hulls retain GJK on CPU until that exact simplex path is ported.
	return minExtent > B3_LINEAR_SLOP && maxExtent <= 16.0f * minExtent;
}

static bool b3MetalSupportsBoxHull( const b3Shape* shape )
{
	if ( b3MetalHasValidHullTopology( shape ) == false ) return false;
	const b3HullData* hull = shape->hull;
	// Restrict the first hull-hull path to the canonical Box3D box allocation and topology.
	// Generic eight-vertex hulls stay on the CPU until the full hull SAT path is ported.
	return hull->vertexCount == 8 && hull->faceCount == 6 && hull->edgeCount == 24 &&
		hull->byteCount == (int32_t)sizeof( b3BoxHull ) && hull->vertexOffset == (int32_t)offsetof( b3BoxHull, boxVertices ) &&
		hull->pointOffset == (int32_t)offsetof( b3BoxHull, boxPoints ) && hull->edgeOffset == (int32_t)offsetof( b3BoxHull, boxEdges ) &&
		hull->planeOffset == (int32_t)offsetof( b3BoxHull, boxPlanes ) && hull->faceOffset == (int32_t)offsetof( b3BoxHull, boxFaces );
}

static bool b3MetalSupportsBoxPair( const b3World* world, const b3Shape* shapeA, const b3Shape* shapeB )
{
	if ( world == NULL || b3MetalSupportsBoxHull( shapeA ) == false || b3MetalSupportsBoxHull( shapeB ) == false ||
		shapeA->bodyId < 0 || shapeA->bodyId >= world->bodies.count || shapeB->bodyId < 0 || shapeB->bodyId >= world->bodies.count )
	{
		return false;
	}
	b3Vec3 extentA = b3Sub( shapeA->hull->aabb.upperBound, shapeA->hull->aabb.lowerBound );
	b3Vec3 extentB = b3Sub( shapeB->hull->aabb.upperBound, shapeB->hull->aabb.lowerBound );
	// High-aspect reference faces remain CPU-owned until the Metal cached-face
	// rebuild matches Box3D's rotating clipped-face acceptance at that scale.
	float minExtentA = b3MinFloat( extentA.x, b3MinFloat( extentA.y, extentA.z ) );
	float maxExtentA = b3MaxFloat( extentA.x, b3MaxFloat( extentA.y, extentA.z ) );
	float minExtentB = b3MinFloat( extentB.x, b3MinFloat( extentB.y, extentB.z ) );
	float maxExtentB = b3MaxFloat( extentB.x, b3MaxFloat( extentB.y, extentB.z ) );
	const b3Body* bodyA = world->bodies.data + shapeA->bodyId;
	const b3Body* bodyB = world->bodies.data + shapeB->bodyId;
	return ( bodyA->type != b3_staticBody || bodyB->type != b3_staticBody ) && minExtentA > B3_LINEAR_SLOP &&
		minExtentB > B3_LINEAR_SLOP && maxExtentA <= 16.0f * minExtentA && maxExtentB <= 16.0f * minExtentB;
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
	uint64_t hullEdgeCount = 0;
	uint64_t hullFaceCount = 0;
	bool valid = true;
	for ( int shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex )
	{
		const b3Shape* shape = world->shapes.data + shapeIndex;
		if ( shape->id != shapeIndex ||
			( b3MetalSupportsHull( shape ) == false && b3MetalSupportsBoxHull( shape ) == false ) )
		{
			continue;
		}
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
		hullEdgeCount += (uint32_t)hull->edgeCount;
		hullFaceCount += (uint32_t)hull->faceCount;
		if ( hullPointCount > UINT32_MAX || hullPlaneCount > UINT32_MAX || hullTriangleCount > UINT32_MAX ||
			 hullEdgeCount > UINT32_MAX || hullFaceCount > UINT32_MAX )
		{
			valid = false;
			break;
		}
	}
	b3HullMap_cleanup( &hullMap );

	if ( valid && ( hullPointCount > NSUIntegerMax / sizeof( b3MetalFloat4 ) ||
					 hullPlaneCount > NSUIntegerMax / sizeof( b3MetalFloat4 ) ||
					 hullTriangleCount > NSUIntegerMax / sizeof( b3MetalHullTriangle ) ||
					 hullEdgeCount > NSUIntegerMax / sizeof( b3MetalHullEdge ) ||
					 hullFaceCount > NSUIntegerMax / sizeof( uint32_t ) ) )
	{
		valid = false;
	}
	NSUInteger hullPointBytes = (NSUInteger)hullPointCount * sizeof( b3MetalFloat4 );
	NSUInteger hullPlaneBytes = (NSUInteger)hullPlaneCount * sizeof( b3MetalFloat4 );
	NSUInteger hullTriangleBytes = (NSUInteger)hullTriangleCount * sizeof( b3MetalHullTriangle );
	NSUInteger hullEdgeBytes = (NSUInteger)hullEdgeCount * sizeof( b3MetalHullEdge );
	NSUInteger hullFaceBytes = (NSUInteger)hullFaceCount * sizeof( uint32_t );
	if ( valid == false || b3MetalEnsureShapeGeometryCapacity( context, shapeCount, hullPointBytes, hullPlaneBytes,
			hullTriangleBytes, hullEdgeBytes, hullFaceBytes ) == false )
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
	b3MetalHullEdge* hullEdges = context->convexHullEdgeBuffer.contents;
	uint32_t* hullFaces = context->convexHullFaceBuffer.contents;
	uint32_t hullPointCursor = 0;
	uint32_t hullPlaneCursor = 0;
	uint32_t hullTriangleCursor = 0;
	uint32_t hullEdgeCursor = 0;
	uint32_t hullFaceCursor = 0;
	for ( int uniqueIndex = 0; uniqueIndex < uniqueCount; ++uniqueIndex )
	{
		const b3HullData* hull = uniqueHulls[uniqueIndex];
		const b3Vec3* points = b3GetHullPoints( hull );
		const b3HullVertex* vertices = b3GetHullVertices( hull );
		const b3Plane* planes = b3GetHullPlanes( hull );
		const b3HullFace* faces = b3GetHullFaces( hull );
		const b3HullHalfEdge* edges = b3GetHullEdges( hull );
		b3MetalShapeGeometry* record = uniqueRecords + uniqueIndex;
		record->type = b3_hullShape;
		record->pointOffset = hullPointCursor;
		record->pointCount = (uint32_t)hull->vertexCount;
		for ( int pointIndex = 0; pointIndex < hull->vertexCount; ++pointIndex )
		{
			hullPoints[hullPointCursor++] = (b3MetalFloat4){
				points[pointIndex].x, points[pointIndex].y, points[pointIndex].z, (float)vertices[pointIndex].edge,
			};
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
		record->edgeOffset = hullEdgeCursor;
		record->edgeCount = (uint32_t)hull->edgeCount;
		for ( int edgeIndex = 0; edgeIndex < hull->edgeCount; ++edgeIndex )
		{
			hullEdges[hullEdgeCursor++] = (b3MetalHullEdge){
				edges[edgeIndex].origin, edges[edgeIndex].twin, edges[edgeIndex].next, edges[edgeIndex].face,
			};
		}
		for ( int faceIndex = 0; faceIndex < hull->faceCount; ++faceIndex )
		{
			hullFaces[hullFaceCursor++] = faces[faceIndex].edge;
		}
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
	B3_ASSERT( hullEdgeCursor == hullEdgeCount );
	B3_ASSERT( hullFaceCursor == hullFaceCount );
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

bool b3MetalSyncBodyStates( const b3MetalContext* context, b3World* world )
{
	if ( context == NULL || world == NULL || context->bodyStateBuffer == nil ) return false;
	b3SolverSet* awakeSet = b3Array_Get( world->solverSets, b3_awakeSet );
	int bodyCount = awakeSet->bodyStates.count;
	if ( bodyCount < 0 || context->bodyStateResidentCount != bodyCount ) return false;
	if ( (NSUInteger)bodyCount > NSUIntegerMax / sizeof( b3BodyState ) ) return false;
	NSUInteger bytes = (NSUInteger)bodyCount * sizeof( b3BodyState );
	if ( bytes > context->bodyStateCapacity ) return false;
	if ( bytes > 0 ) memcpy( awakeSet->bodyStates.data, context->bodyStateBuffer.contents, bytes );
	world->metalBodyStateSyncCount += 1;
	world->metalLastBodyStateReadbackBytes = bytes;
	return true;
}

bool b3MetalSyncBodySims( const b3MetalContext* context, b3World* world )
{
	if ( context == NULL || world == NULL || context->convexBodyTransformBuffer == nil ||
		 context->convexBodyTransformCount != world->bodies.count ||
		 context->convexBodyTransformStepIndex != world->stepIndex ||
		 context->convexBodyTransformRevision != world->metalBodyTransformRevision )
	{
		return false;
	}
	b3SolverSet* awakeSet = b3Array_Get( world->solverSets, b3_awakeSet );
	for ( int simIndex = 0; simIndex < awakeSet->bodySims.count; ++simIndex )
	{
		b3BodySim* sim = awakeSet->bodySims.data + simIndex;
		int bodyId = sim->bodyId;
		b3WorldTransform transform;
		int residentIndex = B3_NULL_INDEX;
		uint32_t stateFlags = 0;
		if ( b3MetalReadResidentBodyTransform( context, world, bodyId, &transform, &residentIndex, &stateFlags ) == false ||
			 residentIndex != simIndex )
		{
			return false;
		}
		sim->transform = transform;
		sim->center = b3TransformWorldPoint( transform, sim->localCenter );
		sim->center0 = sim->center;
		sim->rotation0 = transform.q;
		b3Matrix3 rotationMatrix = b3MakeMatrixFromQuat( transform.q );
		sim->invInertiaWorld =
			b3MulMM( b3MulMM( rotationMatrix, sim->invInertiaLocal ), b3Transpose( rotationMatrix ) );
		sim->force = b3Vec3_zero;
		sim->torque = b3Vec3_zero;
		sim->flags &= ~b3_bodyTransientFlags;
		sim->flags |= stateFlags & ( b3_isSpeedCapped | b3_hadTimeOfImpact );

		b3Body* body = world->bodies.data + bodyId;
		body->bodyMoveIndex = simIndex;
		body->sleepTime = 0.0f;
		body->sleepVelocity =
			( (const b3MetalBodyTransform*)context->convexBodyTransformBuffer.contents )[bodyId].sleepVelocity;
		body->flags &= ~b3_bodyTransientFlags;
		body->flags |= stateFlags & ( b3_isSpeedCapped | b3_hadTimeOfImpact );
	}
	world->metalBodySimSyncCount += 1;
	world->metalLastBodySimSyncCount = (uint64_t)awakeSet->bodySims.count;
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

b3MetalContactInputSeed* b3MetalBeginContactInputBootstrap( b3MetalContext* context, int capacity )
{
	if ( context == NULL || capacity <= 0 || (NSUInteger)capacity > NSUIntegerMax / sizeof( b3MetalContactInputSeed ) )
	{
		return NULL;
	}
	b3MetalCancelVirginContactInputBootstrap( context );
	context->contactInputBootstrapCommitted = false;
	context->contactInputBootstrapPairSeeds = false;
	context->contactInputBootstrapPrivateTopologyCandidate = false;
	context->contactInputBootstrapCount = 0;
	context->contactInputSeedBeginCapacity = 0;
	NSUInteger requiredBytes = (NSUInteger)capacity * sizeof( b3MetalContactInputSeed );
	if ( context->contactInputSeedCapacity < requiredBytes )
	{
		NSUInteger newCapacity = context->contactInputSeedCapacity > 0 ? context->contactInputSeedCapacity : 4096;
		while ( newCapacity < requiredBytes )
		{
			if ( newCapacity > NSUIntegerMax / 2 ) return NULL;
			newCapacity *= 2;
		}
		id<MTLBuffer> buffer = [context->device newBufferWithLength:newCapacity options:MTLResourceStorageModeShared];
		if ( buffer == nil ) return NULL;
		[context->contactInputSeedBuffer release];
		context->contactInputSeedBuffer = buffer;
		context->contactInputSeedCapacity = newCapacity;
	}
	if ( context->contactInputBootstrapStatusBuffer == nil )
	{
		context->contactInputBootstrapStatusBuffer =
			[context->device newBufferWithLength:sizeof( uint32_t ) options:MTLResourceStorageModeShared];
		if ( context->contactInputBootstrapStatusBuffer == nil ) return NULL;
	}
	context->contactInputSeedBeginCapacity = capacity;
	return context->contactInputSeedBuffer.contents;
}

void b3MetalCancelContactInputBootstrap( b3MetalContext* context )
{
	if ( context == NULL ) return;
	context->contactInputSeedBeginCapacity = 0;
	context->contactInputBootstrapCount = 0;
	context->contactInputBootstrapCommitted = false;
	context->contactInputBootstrapPairSeeds = false;
	context->contactInputBootstrapPrivateTopologyCandidate = false;
}

void b3MetalCancelVirginContactInputBootstrap( b3MetalContext* context )
{
	if ( context == NULL ) return;
	context->virginContactInputBootstrapPending = false;
	context->virginContactInputBootstrapCount = 0;
}

void b3MetalCancelPrivateColdContactSchedule( b3MetalContext* context )
{
	if ( context == NULL ) return;
	context->privateColdContactScheduleCount = 0;
	context->privateColdContactScheduleWideCount = 0;
	context->privateColdContactPrepareGeneration = 0;
	context->privateColdContactPairRevision = UINT64_MAX;
	context->privateColdContactGraphRevision = UINT64_MAX;
	context->privateColdContactInputRevision = UINT64_MAX;
}

bool b3MetalHasPrivateColdContactSchedule( const b3MetalContext* context, const b3World* world,
	int* contactCount, int* wideCount )
{
	if ( contactCount != NULL ) *contactCount = 0;
	if ( wideCount != NULL ) *wideCount = 0;
	if ( context == NULL || world == NULL || context->privateColdContactScheduleCount <= 0 ||
		context->privateColdContactScheduleWideCount <= 0 || context->privateColdContactScheduleBuffer == nil ||
		context->privateColdContactPrepareGeneration == 0 ||
		context->privateColdContactPrepareGeneration != context->contactPrepareGeneration ||
		context->privateColdContactPairRevision != world->broadPhase.pairSetRevision ||
		context->privateColdContactGraphRevision != world->constraintGraph.revision ||
		context->privateColdContactInputRevision != world->metalContactInputRevision ||
		context->privateColdContactScheduleCount != world->contacts.count ||
		b3GetIdCount( &world->contactIdPool ) != world->contacts.count ||
		world->broadPhase.pairSet.count != (uint32_t)world->contacts.count )
	{
		return false;
	}
	if ( contactCount != NULL ) *contactCount = context->privateColdContactScheduleCount;
	if ( wideCount != NULL ) *wideCount = context->privateColdContactScheduleWideCount;
	return true;
}

bool b3MetalCommitDeferredContactTopology( b3MetalContext* context, const b3World* world, int contactCount )
{
	if ( context == NULL || world == NULL || contactCount <= 0 ||
		context->privateColdContactScheduleCount != contactCount ||
		context->privateColdContactPrepareGeneration == 0 ||
		context->privateColdContactPrepareGeneration != context->contactPrepareGeneration ||
		context->privateColdContactPairRevision != world->broadPhase.pairSetRevision ||
		context->privateColdContactInputRevision != world->metalContactInputRevision ||
		// Materialization legitimately advances the graph revision, so only a
		// regression (stale schedule newer than the world) fails here.
		context->privateColdContactGraphRevision > world->constraintGraph.revision ||
		context->convexManifoldInputsPrivate == false || context->convexManifoldInputCount != contactCount ||
		context->convexManifoldCandidateCount != contactCount ||
		context->convexManifoldInputPairRevision != world->broadPhase.pairSetRevision ||
		context->convexManifoldInputRevision != world->metalContactInputRevision ||
		context->convexManifoldTableCount != world->contacts.count || world->contacts.count != contactCount ||
		b3GetIdCount( &world->contactIdPool ) != contactCount )
	{
		b3MetalCancelPrivateColdContactSchedule( context );
		return false;
	}
	context->convexManifoldInputGraphRevision = world->constraintGraph.revision;
	b3MetalCancelPrivateColdContactSchedule( context );
	return true;
}

bool b3MetalHasVirginContactInputBootstrap( const b3MetalContext* context, int contactCount )
{
	return context != NULL && contactCount > 0 && context->virginContactInputBootstrapPending &&
		context->virginContactInputBootstrapCount == contactCount && context->pairContactSeedBuffer != nil &&
		(NSUInteger)contactCount <= context->pairContactSeedCapacity / sizeof( b3MetalPairContactSeed );
}

bool b3MetalCommitVirginContactInputBootstrap( b3MetalContext* context, const b3World* world, int contactCount )
{
	if ( context == NULL || world == NULL || contactCount <= 0 ||
		b3MetalHasVirginContactInputBootstrap( context, contactCount ) == false ||
		world->contacts.count != contactCount || b3GetIdCount( &world->contactIdPool ) != contactCount ||
		world->broadPhase.pairSet.count != (uint32_t)contactCount ||
		world->broadPhase.pairSetRevision != context->virginContactInputBootstrapPairRevision + (uint64_t)contactCount ||
		world->constraintGraph.revision != context->virginContactInputBootstrapGraphRevision ||
		world->metalContactInputRevision != context->virginContactInputBootstrapRevision + (uint64_t)contactCount ||
		world->metalPairShapeRevision != context->virginContactInputBootstrapShapeRevision ||
		world->metalDefaultFrictionCallback == false || world->metalDefaultRestitutionCallback == false ||
		world->contactRecycleDistance != 0.0f || world->recording != NULL )
	{
		b3MetalCancelVirginContactInputBootstrap( context );
		return false;
	}
	const b3SolverSet* awakeSet = world->solverSets.data + b3_awakeSet;
	if ( awakeSet->contactIndices.count != contactCount )
	{
		b3MetalCancelVirginContactInputBootstrap( context );
		return false;
	}
	for ( int colorIndex = 0; colorIndex < B3_GRAPH_COLOR_COUNT; ++colorIndex )
	{
		const b3GraphColor* color = world->constraintGraph.colors + colorIndex;
		if ( color->convexContacts.count != 0 || color->contacts.count != 0 )
		{
			b3MetalCancelVirginContactInputBootstrap( context );
			return false;
		}
	}
	const b3MetalPairContactSeed* seeds = context->pairContactSeedBuffer.contents;
	bool privateTopologyCandidate = true;
	for ( int contactId = 0; contactId < contactCount; ++contactId )
	{
		const b3MetalPairContactSeed* seed = seeds + contactId;
		if ( seed->shapeIndexA < 0 || seed->shapeIndexA >= world->shapes.count || seed->shapeIndexB < 0 ||
			seed->shapeIndexB >= world->shapes.count || awakeSet->contactIndices.data[contactId] != contactId )
		{
			b3MetalCancelVirginContactInputBootstrap( context );
			return false;
		}
		const b3Shape* seedShapeA = world->shapes.data + seed->shapeIndexA;
		const b3Shape* seedShapeB = world->shapes.data + seed->shapeIndexB;
		if ( seedShapeA->bodyId < 0 || seedShapeA->bodyId >= world->bodies.count || seedShapeB->bodyId < 0 ||
			 seedShapeB->bodyId >= world->bodies.count )
		{
			b3MetalCancelVirginContactInputBootstrap( context );
			return false;
		}
		bool dynamicA = world->bodies.data[seedShapeA->bodyId].type == b3_dynamicBody;
		bool dynamicB = world->bodies.data[seedShapeB->bodyId].type == b3_dynamicBody;
		bool staticA = world->bodies.data[seedShapeA->bodyId].type == b3_staticBody;
		bool staticB = world->bodies.data[seedShapeB->bodyId].type == b3_staticBody;
		privateTopologyCandidate = privateTopologyCandidate && ( ( dynamicA && staticB ) || ( staticA && dynamicB ) );
		const b3Shape* shapeA = seedShapeA;
		const b3Shape* shapeB = seedShapeB;
		if ( shapeA->type > shapeB->type )
		{
			const b3Shape* swap = shapeA;
			shapeA = shapeB;
			shapeB = swap;
		}
		const b3Contact* contact = world->contacts.data + contactId;
		const uint32_t eventMask = b3_enableContactEvents | b3_enableHitEvents | b3_enablePreSolveEvents;
		bool eligible = ( shapeA->type == b3_sphereShape && shapeB->type == b3_sphereShape ) ||
			( shapeA->type == b3_capsuleShape && shapeB->type == b3_sphereShape ) ||
			( shapeA->type == b3_capsuleShape && shapeB->type == b3_capsuleShape ) ||
			b3MetalSupportsHullSphere( shapeA, shapeB ) || b3MetalSupportsBoxPair( world, shapeA, shapeB );
		if ( contact->contactId != contactId || contact->generation != 1 || contact->shapeIdA != shapeA->id ||
			contact->shapeIdB != shapeB->id || contact->setIndex != b3_awakeSet || contact->colorIndex != B3_NULL_INDEX ||
			contact->localIndex != contactId || contact->manifoldCount != 0 || contact->manifolds != NULL ||
			( contact->flags & ( b3_simTouchingFlag | b3_simEnablePreSolveEvents ) ) != 0 || eligible == false ||
			( shapeA->flags & eventMask ) != 0 || ( shapeB->flags & eventMask ) != 0 )
		{
			b3MetalCancelVirginContactInputBootstrap( context );
			return false;
		}
	}
	if ( context->contactInputBootstrapStatusBuffer == nil )
	{
		context->contactInputBootstrapStatusBuffer =
			[context->device newBufferWithLength:sizeof( uint32_t ) options:MTLResourceStorageModeShared];
		if ( context->contactInputBootstrapStatusBuffer == nil )
		{
			b3MetalCancelVirginContactInputBootstrap( context );
			return false;
		}
	}
	context->contactInputBootstrapCount = contactCount;
	context->contactInputBootstrapPairRevision = world->broadPhase.pairSetRevision;
	context->contactInputBootstrapGraphRevision = world->constraintGraph.revision;
	context->contactInputBootstrapRevision = world->metalContactInputRevision;
	context->contactInputBootstrapCommitted = true;
	context->contactInputBootstrapPairSeeds = true;
	context->contactInputBootstrapPrivateTopologyCandidate = privateTopologyCandidate;
	b3MetalCancelVirginContactInputBootstrap( context );
	return true;
}

bool b3MetalCommitContactInputBootstrap( b3MetalContext* context, const b3World* world, int count )
{
	if ( context == NULL || world == NULL || count <= 0 || context->contactInputSeedBuffer == nil ||
		count > context->contactInputSeedBeginCapacity ||
		world->metalDefaultFrictionCallback == false || world->metalDefaultRestitutionCallback == false ||
		world->recording != NULL )
	{
		b3MetalCancelContactInputBootstrap( context );
		return false;
	}
	b3SolverSet* awakeSet = world->solverSets.data + b3_awakeSet;
	if ( awakeSet->contactIndices.count != count )
	{
		b3MetalCancelContactInputBootstrap( context );
		return false;
	}
	for ( int colorIndex = 0; colorIndex < B3_GRAPH_COLOR_COUNT; ++colorIndex )
	{
		const b3GraphColor* color = world->constraintGraph.colors + colorIndex;
		if ( color->convexContacts.count != 0 || color->contacts.count != 0 )
		{
			b3MetalCancelContactInputBootstrap( context );
			return false;
		}
	}
	const b3MetalContactInputSeed* seeds = context->contactInputSeedBuffer.contents;
	bool valid = true;
	for ( int i = 0; i < count; ++i )
	{
		const b3MetalContactInputSeed* seed = seeds + i;
		if ( seed->contactId >= (uint32_t)world->contacts.count ||
			awakeSet->contactIndices.data[i] != (int)seed->contactId )
		{
			valid = false;
			break;
		}
		const b3Contact* contact = world->contacts.data + seed->contactId;
		if ( contact->contactId != (int)seed->contactId || contact->generation != seed->generation ||
			contact->shapeIdA != (int)seed->shapeIdA || contact->shapeIdB != (int)seed->shapeIdB ||
			contact->setIndex != b3_awakeSet || contact->colorIndex != B3_NULL_INDEX || contact->localIndex != i ||
			contact->manifoldCount != 0 ||
			( contact->flags & ( b3_simTouchingFlag | b3_simEnablePreSolveEvents ) ) != 0 ||
			seed->shapeIdA >= (uint32_t)world->shapes.count || seed->shapeIdB >= (uint32_t)world->shapes.count )
		{
			valid = false;
			break;
		}
		const b3Shape* shapeA = world->shapes.data + seed->shapeIdA;
		const b3Shape* shapeB = world->shapes.data + seed->shapeIdB;
		if ( shapeA->id != (int)seed->shapeIdA || shapeB->id != (int)seed->shapeIdB )
		{
			valid = false;
			break;
		}
		bool eligible = ( shapeA->type == b3_sphereShape && shapeB->type == b3_sphereShape ) ||
			( shapeA->type == b3_capsuleShape && shapeB->type == b3_sphereShape ) ||
			( shapeA->type == b3_capsuleShape && shapeB->type == b3_capsuleShape ) ||
			b3MetalSupportsHullSphere( shapeA, shapeB ) || b3MetalSupportsBoxPair( world, shapeA, shapeB );
		const uint32_t eventMask = b3_enableContactEvents | b3_enableHitEvents | b3_enablePreSolveEvents;
		if ( eligible == false || ( shapeA->flags & eventMask ) != 0 || ( shapeB->flags & eventMask ) != 0 )
		{
			valid = false;
			break;
		}
	}
	if ( valid == false )
	{
		b3MetalCancelContactInputBootstrap( context );
		return false;
	}
	context->contactInputSeedBeginCapacity = 0;
	context->contactInputBootstrapCount = count;
	context->contactInputBootstrapPairRevision = world->broadPhase.pairSetRevision;
	context->contactInputBootstrapGraphRevision = world->constraintGraph.revision;
	context->contactInputBootstrapRevision = world->metalContactInputRevision;
	context->contactInputBootstrapCommitted = true;
	context->contactInputBootstrapPairSeeds = false;
	context->contactInputBootstrapPrivateTopologyCandidate = false;
	return true;
}

bool b3MetalCanBootstrapConvexManifoldInputs( const b3MetalContext* context, const b3World* world, int contactCount )
{
	return context != NULL && world != NULL && context->contactInputBootstrapCommitted && contactCount > 0 &&
		context->contactInputBootstrapCount == contactCount &&
		( context->contactInputBootstrapPairSeeds ? context->pairContactSeedBuffer != nil :
			context->contactInputSeedBuffer != nil ) &&
		context->contactInputBootstrapStatusBuffer != nil &&
		context->contactInputBootstrapPairRevision == world->broadPhase.pairSetRevision &&
		context->contactInputBootstrapGraphRevision == world->constraintGraph.revision &&
		context->contactInputBootstrapRevision == world->metalContactInputRevision;
}

bool b3MetalCanReuseConvexManifoldInputs( const b3MetalContext* context, const b3World* world, int contactCount )
{
	return context != NULL && world != NULL && contactCount > 0 &&
		( context->convexManifoldInputsPrivate ? context->convexManifoldPrivateInputBuffer != nil :
			context->convexManifoldInputBuffer != nil ) &&
		context->convexManifoldInputCount == contactCount &&
		context->convexManifoldInputPairRevision == world->broadPhase.pairSetRevision &&
		context->convexManifoldInputGraphRevision == world->constraintGraph.revision &&
		context->convexManifoldInputRevision == world->metalContactInputRevision;
}

// Narrow-phase encode half (Phase-1 split): input validation, registry setup,
// and kernel encoding into a fresh command buffer, ending encoding WITHOUT
// committing. The caller commits (alone, or merged with the solve buffer) and
// finishes with b3MetalConsumeConvexManifolds. Returns true with
// *bufferOut == nil when no dispatch was needed (empty input); false on any
// validation or allocation failure. Behavior matches the pre-split function.
static bool b3MetalEncodeConvexManifolds( b3MetalContext* context, const b3World* world, const int* contactIndices,
	int contactCount, bool collectBypass, b3MetalNarrowEncode* encodeOut, id<MTLCommandBuffer>* bufferOut )
{
	if ( context != NULL ) context->contactTransitionCount = 0;
	if ( encodeOut != NULL ) *encodeOut = (b3MetalNarrowEncode){ 0 };
	if ( bufferOut != NULL ) *bufferOut = nil;
	if ( context == NULL || world == NULL || contactCount < 0 || encodeOut == NULL || bufferOut == NULL )
	{
		return false;
	}
	bool reuseInputs = contactIndices == NULL && b3MetalCanReuseConvexManifoldInputs( context, world, contactCount );
	bool bootstrapInputs = contactIndices == NULL && reuseInputs == false &&
		b3MetalCanBootstrapConvexManifoldInputs( context, world, contactCount );
	bool pairSeedBootstrapInputs = bootstrapInputs && context->contactInputBootstrapPairSeeds;
	// Private-topology eligibility predicate (GPU site). Keep in sync with the
	// host admission in b3Collide (physics_world.c) and the solver gate in
	// b3TrySolvePrivateColdContacts (solver.c); intentionally not merged.
	bool privateColdTopology = pairSeedBootstrapInputs && context->contactInputBootstrapPrivateTopologyCandidate &&
		world->metalDeferredContactTopologyPending == false && world->metalFinalizationEnabled &&
		world->metalBroadPhaseEnabled && world->enableSleep == false && world->enableContinuous == false &&
		world->splitIslandId == B3_NULL_INDEX && world->sensors.count == 0 &&
		b3GetIdCount( &world->jointIdPool ) == 0 && world->contactRecycleDistance == 0.0f &&
		world->recording == NULL && world->preSolveFcn == NULL && world->customFilterFcn == NULL &&
		world->metalDefaultFrictionCallback && world->metalDefaultRestitutionCallback &&
		contactCount >= world->metalMinimumBodyCount &&
		world->solverSets.data[b3_awakeSet].bodyStates.count >= world->metalMinimumBodyCount;
	b3MetalCancelPrivateColdContactSchedule( context );
	if ( contactIndices == NULL && reuseInputs == false && bootstrapInputs == false ) return false;
	if ( contactIndices != NULL ) b3MetalCancelContactInputBootstrap( context );
	if ( bootstrapInputs )
	{
		// Consume this one-shot authority before allocating or dispatching. Any
		// failure below must force the caller onto the legacy CPU pack rather
		// than retrying a seed set whose device side effects are uncertain.
		b3MetalCancelContactInputBootstrap( context );
		context->convexManifoldInputCount = 0;
		context->convexManifoldCandidateCount = 0;
		context->convexManifoldInputPairRevision = UINT64_MAX;
		context->convexManifoldInputGraphRevision = UINT64_MAX;
		context->convexManifoldInputRevision = UINT64_MAX;
	}
	( (b3World*)world )->metalLastNarrowPhaseResultCount = 0;
	( (b3World*)world )->metalLastNarrowPhaseManifoldTableCount = 0;
	if ( contactCount == 0 )
	{
		context->convexManifoldTableCount = 0;
		return true;
	}
	if ( (NSUInteger)contactCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldInput ) ||
		 (NSUInteger)contactCount > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) ||
		 world->contacts.count < 0 || (NSUInteger)world->contacts.count > NSUIntegerMax / sizeof( b3MetalConvexManifoldResult ) ||
		 (NSUInteger)world->contacts.count > NSUIntegerMax / sizeof( b3MetalContactTransition ) )
	{
		return false;
	}
	@autoreleasepool
	{
		int candidateCount = reuseInputs ? context->convexManifoldCandidateCount : bootstrapInputs ? contactCount : 0;
		int hitEventContactCount = reuseInputs ? context->contactHitEventIdCount : 0;
		if ( reuseInputs == false && bootstrapInputs == false )
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
					b3MetalSupportsHullSphere( shapeA, shapeB ) || b3MetalSupportsBoxPair( world, shapeA, shapeB );
				candidateCount += eligible;
				hitEventContactCount += eligible && ( ( shapeA->flags & b3_enableHitEvents ) != 0 ||
					( shapeB->flags & b3_enableHitEvents ) != 0 );
			}
		}
		if ( reuseInputs == false ) context->contactHitEventIdCount = 0;
		if ( candidateCount == 0 )
		{
			context->convexManifoldTableCount = 0;
			return true;
		}

		NSUInteger inputBytes = (NSUInteger)contactCount * sizeof( b3MetalConvexManifoldInput );
		NSUInteger resultBytes = (NSUInteger)contactCount * sizeof( b3MetalConvexManifoldResult );
		uint32_t blockCount = ( (uint32_t)contactCount + 255u ) / 256u;
		NSUInteger blockBytes = (NSUInteger)blockCount * sizeof( b3MetalManifoldBlock );
		NSUInteger tableCount = world->contacts.count > 0 ? (NSUInteger)world->contacts.count : 1;
		NSUInteger transitionBytes = privateColdTopology ? 0 : tableCount * sizeof( b3MetalContactTransition );
		NSUInteger privateScheduleWideCount = privateColdTopology
			? ( (NSUInteger)contactCount + B3_SIMD_WIDTH - 1 ) / B3_SIMD_WIDTH : 0;
		NSUInteger privateScheduleBytes = privateScheduleWideCount * B3_SIMD_WIDTH * sizeof( uint32_t );
		NSUInteger bodyOwnerBytes = privateColdTopology
			? (NSUInteger)world->bodies.count * sizeof( uint32_t ) : 0;
		NSUInteger tableBytes = tableCount * sizeof( b3MetalConvexManifoldResult );
		if ( tableCount > NSUIntegerMax / sizeof( b3MetalContactPrepareInput ) ||
			 tableCount > NSUIntegerMax / sizeof( b3MetalContactImpulseResult ) ) return false;
		NSUInteger prepareTableBytes = tableCount * sizeof( b3MetalContactPrepareInput );
		NSUInteger impulseTableBytes = tableCount * sizeof( b3MetalContactImpulseResult );
		NSUInteger previousImpulseCapacity = context->contactImpulseResultCapacity;
		if ( b3MetalEnsureShapeGeometryRegistry( context, world ) == false ||
			 b3MetalEnsureBodyTransformRegistry( context, world ) == false ||
			 b3MetalEnsureConvexManifoldCapacity( context, inputBytes, resultBytes, resultBytes, transitionBytes, blockBytes, tableBytes,
				 bootstrapInputs || ( reuseInputs && context->convexManifoldInputsPrivate ) ) == false ||
			 b3MetalEnsureContactPrepareTableCapacity( context, prepareTableBytes ) == false ||
			 ( privateColdTopology && b3MetalEnsurePrivateColdTopologyCapacity(
			   context, privateScheduleBytes, bodyOwnerBytes ) == false ) ||
			 b3MetalEnsureContactImpulseResultCapacity( context, impulseTableBytes ) == false ||
			 b3MetalEnsureContactHitEventIdCapacity( context, (NSUInteger)hitEventContactCount * sizeof( int ) ) == false )
		{
			return false;
		}
		if ( privateColdTopology && ( privateScheduleBytes == 0 || bodyOwnerBytes == 0 ||
			 context->privateColdContactScheduleBuffer == nil || context->privateColdBodyOwnerBuffer == nil ) )
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
		id<MTLBuffer> manifoldInputBuffer = bootstrapInputs || ( reuseInputs && context->convexManifoldInputsPrivate )
			? context->convexManifoldPrivateInputBuffer : context->convexManifoldInputBuffer;
		b3MetalConvexManifoldInput* inputs = reuseInputs == false && bootstrapInputs == false
			? context->convexManifoldInputBuffer.contents : NULL;
		if ( reuseInputs == false && bootstrapInputs == false )
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
				b3MetalConvexManifoldInput* input = inputs + i;
				input->contactId = (uint32_t)contactIndex;
				input->contactGeneration = contact->generation;
				if ( shapeA->type == b3_hullShape && shapeB->type == b3_hullShape )
				{
					const b3SATCache* satCache = &contact->convexContact.cache.satCache;
					input->satSeparation = satCache->separation;
					input->satCache = (uint32_t)satCache->type | ( (uint32_t)satCache->indexA << 8 ) |
						( (uint32_t)satCache->indexB << 16 ) | ( (uint32_t)satCache->hit << 24 );
				}
				bool eligible = ( shapeA->type == b3_sphereShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_sphereShape ) ||
					( shapeA->type == b3_capsuleShape && shapeB->type == b3_capsuleShape ) ||
					b3MetalSupportsHullSphere( shapeA, shapeB ) || b3MetalSupportsBoxPair( world, shapeA, shapeB );
				if ( eligible == false ) continue;
				if ( ( shapeA->flags & b3_enableHitEvents ) != 0 || ( shapeB->flags & b3_enableHitEvents ) != 0 )
				{
					hitEventContactIds[context->contactHitEventIdCount++] = contactIndex;
				}
				input->eligible = 1;
				input->shapeIdA = (uint32_t)contact->shapeIdA;
				input->shapeIdB = (uint32_t)contact->shapeIdB;
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
					isFast == false && ( shapeA->flags & b3_enableHitEvents ) == 0 &&
					( shapeB->flags & b3_enableHitEvents ) == 0 && world->recording == NULL;
				input->prepareEligible |= bypassEligible ? 2u : 0u;
			}
			context->convexManifoldInputCount = contactCount;
			context->convexManifoldCandidateCount = candidateCount;
			context->convexManifoldInputPairRevision = world->broadPhase.pairSetRevision;
			context->convexManifoldInputGraphRevision = world->constraintGraph.revision;
			context->convexManifoldInputRevision = world->metalContactInputRevision;
			context->convexManifoldInputsPrivate = false;
			( (b3World*)world )->metalContactInputPackCount += 1;
			( (b3World*)world )->metalLastContactInputBytes = inputBytes;
		}
		else if ( reuseInputs )
		{
			( (b3World*)world )->metalContactInputReuseCount += 1;
			( (b3World*)world )->metalLastContactInputBytes = 0;
		}
		else
		{
			( (b3World*)world )->metalLastContactInputBytes = 0;
		}

		struct { uint32_t contactCount; float linearSlop, speculativeDistance; uint32_t bodyCount, previousTableCount; } params = {
			(uint32_t)contactCount, B3_LINEAR_SLOP, B3_SPECULATIVE_DISTANCE, (uint32_t)context->convexBodyTransformCount,
			(uint32_t)context->convexManifoldTableCount,
		};
		struct
		{
			uint32_t contactCount, blockCount, previousCount, previousGeneration;
			uint32_t currentGeneration, padding0, padding1, padding2;
		} compactParams = {
			(uint32_t)contactCount, blockCount, (uint32_t)context->contactImpulseResultCount,
			context->contactImpulseResultGeneration, context->contactPrepareGeneration,
			collectBypass ? 1u : 0u, privateColdTopology ? 2u : bootstrapInputs ? 1u : 0u,
			(uint32_t)tableCount,
		};
		NSUInteger scanWidth = context->convexManifoldScanPipeline.threadExecutionWidth;
		if ( context->convexManifoldScanPipeline.maxTotalThreadsPerThreadgroup < 256 || scanWidth == 0 ||
			 256 % scanWidth != 0 || 256 / scanWidth > 32 )
		{
			return false;
		}
		id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
		if ( commandBuffer == nil ) return false;
		double encodeStartMs = b3MetalMonotonicMs();
		if ( bootstrapInputs )
		{
			id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
			if ( blitEncoder == nil ) return false;
			if ( privateColdTopology )
			{
				[blitEncoder fillBuffer:context->privateColdContactScheduleBuffer
					range:NSMakeRange( 0, privateScheduleBytes ) value:0xff];
				[blitEncoder fillBuffer:context->privateColdBodyOwnerBuffer
					range:NSMakeRange( 0, bodyOwnerBytes ) value:0];
			}
			else
			{
				[blitEncoder fillBuffer:context->contactTransitionBuffer
					range:NSMakeRange( 0, transitionBytes ) value:0];
			}
			[blitEncoder endEncoding];
		}
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( encoder == nil ) return false;
		if ( bootstrapInputs )
		{
			*( (uint32_t*)context->contactInputBootstrapStatusBuffer.contents ) = 0;
			struct
			{
				uint32_t count, contactCapacity, shapeCount, bodyCount;
				uint32_t prepareEligible, padding0, padding1, padding2;
			} bootstrapParams = {
				(uint32_t)contactCount, (uint32_t)world->contacts.count, (uint32_t)context->convexShapeGeometryCount,
				(uint32_t)context->convexBodyTransformCount, world->contactRecycleDistance == 0.0f ? 1u : 0u,
				privateColdTopology ? 1u : 0u, 0u, 0u,
			};
			_Static_assert( sizeof( bootstrapParams ) == 32, "Metal contact-input-bootstrap parameter ABI changed" );
			id<MTLComputePipelineState> bootstrapPipeline = pairSeedBootstrapInputs
				? context->pairSeedInputBootstrapPipeline : context->contactInputBootstrapPipeline;
			id<MTLBuffer> bootstrapSeedBuffer = pairSeedBootstrapInputs
				? context->pairContactSeedBuffer : context->contactInputSeedBuffer;
			[encoder setComputePipelineState:bootstrapPipeline];
			[encoder setBuffer:bootstrapSeedBuffer offset:0 atIndex:0];
			[encoder setBuffer:manifoldInputBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->convexShapeGeometryBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->convexBodyTransformBuffer offset:0 atIndex:3];
			[encoder setBuffer:context->contactInputBootstrapStatusBuffer offset:0 atIndex:4];
			[encoder setBytes:&bootstrapParams length:sizeof( bootstrapParams ) atIndex:5];
			[encoder setBuffer:privateColdTopology ? context->privateColdBodyOwnerBuffer : context->convexManifoldBlockBuffer
				offset:0 atIndex:6];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)contactCount, 1, 1 )
				threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( bootstrapPipeline ), 1, 1 )];
			[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
		}
		id<MTLComputePipelineState> pipeline = context->convexManifoldPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:manifoldInputBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldResultBuffer offset:0 atIndex:1];
		[encoder setBytes:&params length:sizeof( params ) atIndex:2];
		[encoder setBuffer:context->convexHullPointBuffer offset:0 atIndex:3];
		[encoder setBuffer:context->convexHullPlaneBuffer offset:0 atIndex:4];
		[encoder setBuffer:context->convexHullTriangleBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->convexShapeGeometryBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->convexBodyTransformBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->convexHullEdgeBuffer offset:0 atIndex:8];
		[encoder setBuffer:context->convexHullFaceBuffer offset:0 atIndex:9];
		[encoder setBuffer:context->convexManifoldTableBuffer offset:0 atIndex:10];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)contactCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pipeline ), 1, 1 )];
		[encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

		pipeline = context->convexManifoldScanPipeline;
		[encoder setComputePipelineState:pipeline];
		[encoder setBuffer:context->convexManifoldResultBuffer offset:0 atIndex:0];
		[encoder setBuffer:context->convexManifoldBlockBuffer offset:0 atIndex:1];
		[encoder setBuffer:manifoldInputBuffer offset:0 atIndex:2];
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
		[encoder setBuffer:manifoldInputBuffer offset:0 atIndex:3];
		[encoder setBuffer:context->convexShapeGeometryBuffer offset:0 atIndex:4];
		[encoder setBuffer:context->convexBodyTransformBuffer offset:0 atIndex:5];
		[encoder setBuffer:context->convexManifoldTableBuffer offset:0 atIndex:6];
		[encoder setBuffer:context->contactImpulseResultBuffer offset:0 atIndex:7];
		[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:8];
		[encoder setBuffer:privateColdTopology ? context->privateColdContactScheduleBuffer : context->contactTransitionBuffer
			offset:0 atIndex:9];
		[encoder setBytes:&compactParams length:sizeof( compactParams ) atIndex:10];
		[encoder dispatchThreads:MTLSizeMake( (NSUInteger)contactCount, 1, 1 )
			threadsPerThreadgroup:MTLSizeMake( b3MetalThreadgroupWidth( pipeline ), 1, 1 )];
		[encoder endEncoding];
		encodeOut->bootstrapInputs = bootstrapInputs;
		encodeOut->pairSeedBootstrapInputs = pairSeedBootstrapInputs;
		encodeOut->privateColdTopology = privateColdTopology;
		encodeOut->contactCount = contactCount;
		encodeOut->candidateCount = candidateCount;
		encodeOut->privateScheduleBytes = (uint64_t)privateScheduleBytes;
		encodeOut->privateScheduleWideCount = (uint64_t)privateScheduleWideCount;
		encodeOut->encodeMs = b3MetalMonotonicMs() - encodeStartMs;
		// Manual retain-counting file: the buffer is autoreleased and its
		// pool drains on return, so transfer +1 ownership to the caller,
		// which must release after commit/wait (or via Cancel/destroy).
		*bufferOut = [commandBuffer retain];
		return true;
	}
}

// Narrow-phase consume half (Phase-1 split): validates and publishes the
// results of a command buffer produced by b3MetalEncodeConvexManifolds after
// the caller commits and waits. encodeMs/waitMs are recorded into stats;
// gpuMsOverride >= 0 replaces the buffer's measured GPU time (merged-buffer
// GPU-time split), otherwise the buffer time is used as before.
static bool b3MetalConsumeConvexManifolds( b3MetalContext* context, const b3World* world,
	id<MTLCommandBuffer> commandBuffer, const b3MetalNarrowEncode* encode, double encodeMs, double waitMs,
	double gpuMsOverride, const b3MetalConvexManifoldResult** resultsOut, int* resultCountOut,
	int* residentBypassCountOut, const b3MetalContactTransition** transitionsOut, int* transitionCountOut,
	b3MetalDispatchStats* stats, const char* traceSite )
{
	bool bootstrapInputs = encode->bootstrapInputs;
	bool pairSeedBootstrapInputs = encode->pairSeedBootstrapInputs;
	bool privateColdTopology = encode->privateColdTopology;
	int contactCount = encode->contactCount;
	int candidateCount = encode->candidateCount;
	uint64_t privateScheduleBytes = encode->privateScheduleBytes;
	uint64_t privateScheduleWideCount = encode->privateScheduleWideCount;
	@autoreleasepool
	{
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			context->convexManifoldTableCount = 0;
			if ( bootstrapInputs ) b3MetalCancelContactInputBootstrap( context );
			return false;
		}
		if ( bootstrapInputs &&
			*( (const uint32_t*)context->contactInputBootstrapStatusBuffer.contents ) != 0 )
		{
			context->convexManifoldInputCount = 0;
			context->convexManifoldTableCount = 0;
			b3MetalCancelContactInputBootstrap( context );
			return false;
		}
		const b3MetalManifoldSummary* summary = context->convexManifoldSummaryBuffer.contents;
		uint64_t resultLimit = residentBypassCountOut != NULL ? (uint64_t)contactCount : (uint64_t)candidateCount;
		uint64_t classifiedCount = summary->exceptionCount + summary->transitionCount + summary->stableCount + summary->silentCount;
		bool invalidSummary = summary->errorFlags != 0 || summary->exceptionCount > resultLimit ||
			summary->exceptionCount > INT32_MAX || summary->transitionCount > (uint32_t)contactCount ||
			summary->stableCount > (uint32_t)candidateCount || summary->persistenceMatches > 4ull * summary->stableCount;
		if ( bootstrapInputs )
		{
			invalidSummary = invalidSummary || summary->stableCount != 0 || classifiedCount != (uint64_t)contactCount;
			if ( privateColdTopology )
			{
				invalidSummary = invalidSummary || summary->exceptionCount != 0 ||
					summary->transitionCount != (uint32_t)contactCount || summary->silentCount != 0;
			}
		}
		else
		{
			invalidSummary = invalidSummary || summary->transitionCount != 0 || summary->silentCount != 0 ||
				( residentBypassCountOut != NULL && summary->exceptionCount + summary->stableCount != (uint64_t)contactCount );
		}
		if ( invalidSummary )
		{
			context->convexManifoldTableCount = 0;
			if ( bootstrapInputs ) b3MetalCancelContactInputBootstrap( context );
			return false;
		}
		int resultCount = (int)summary->exceptionCount;
		int transitionCount = (int)summary->transitionCount;
		if ( privateColdTopology == false && transitionCount > 0 && transitionsOut == NULL )
		{
			context->convexManifoldTableCount = 0;
			return false;
		}
		int residentBypassCount = residentBypassCountOut != NULL ? (int)summary->stableCount : 0;
		const b3MetalConvexManifoldResult* completedResults = context->convexManifoldCompactBuffer.contents;
		const b3MetalContactTransition* completedTransitions = context->contactTransitionBuffer.contents;
		uint64_t persistenceMatchCount = summary->persistenceMatches;
		uint64_t prepareDeviceRefreshCount = (uint64_t)residentBypassCount + (uint64_t)transitionCount;
		for ( int i = 0; i < resultCount; ++i )
		{
			uint32_t persistedBits = completedResults[i].persistedBits;
			persistenceMatchCount += ( persistedBits & 1u ) + ( ( persistedBits >> 1 ) & 1u ) +
				( ( persistedBits >> 2 ) & 1u ) + ( ( persistedBits >> 3 ) & 1u );
			prepareDeviceRefreshCount += ( completedResults[i].residentFlags & 2u ) != 0;
		}
		int seenTransitions = 0;
		if ( bootstrapInputs && privateColdTopology == false )
		{
			for ( int contactId = 0; contactId < world->contacts.count; ++contactId )
			{
				const b3MetalContactTransition* transition = completedTransitions + contactId;
				if ( transition->pointCount == 0 )
				{
					if ( transition->contactGeneration != 0 )
					{
						context->convexManifoldTableCount = 0;
						return false;
					}
					continue;
				}
				if ( transition->contactGeneration == 0 || transition->pointCount > B3_MAX_MANIFOLD_POINTS )
				{
					context->convexManifoldTableCount = 0;
					return false;
				}
				const b3Contact* contact = world->contacts.data + contactId;
				const b3MetalContactPrepareInput* prepare =
					( (const b3MetalContactPrepareInput*)context->contactPrepareTableBuffer.contents ) + contactId;
				if ( contact->contactId != contactId || contact->generation != transition->contactGeneration ||
					prepare->contactId != (uint32_t)contactId ||
					prepare->contactGeneration != transition->contactGeneration ||
					prepare->generation != context->contactPrepareGeneration || prepare->manifold != 0 )
				{
					context->convexManifoldTableCount = 0;
					return false;
				}
				seenTransitions += 1;
			}
		}
		if ( privateColdTopology == false && seenTransitions != transitionCount )
		{
			context->convexManifoldTableCount = 0;
			return false;
		}
		( (b3World*)world )->metalContactPersistenceMatchCount += persistenceMatchCount;
		( (b3World*)world )->metalContactPrepareDeviceRefreshCount += prepareDeviceRefreshCount;
		context->convexManifoldTableCount = world->contacts.count;
		if ( privateColdTopology )
		{
			context->privateColdContactScheduleCount = transitionCount;
			context->privateColdContactScheduleWideCount = (int)privateScheduleWideCount;
			context->privateColdContactPrepareGeneration = context->contactPrepareGeneration;
			context->privateColdContactPairRevision = world->broadPhase.pairSetRevision;
			context->privateColdContactGraphRevision = world->constraintGraph.revision;
			context->privateColdContactInputRevision = world->metalContactInputRevision;
		}
		if ( bootstrapInputs )
		{
			context->convexManifoldInputCount = contactCount;
			context->convexManifoldCandidateCount = contactCount;
			context->convexManifoldInputPairRevision = world->broadPhase.pairSetRevision;
			context->convexManifoldInputGraphRevision = world->constraintGraph.revision;
			context->convexManifoldInputRevision = world->metalContactInputRevision;
			context->convexManifoldInputsPrivate = true;
			b3MetalCancelContactInputBootstrap( context );
			( (b3World*)world )->metalLastContactInputBytes = 0;
		}
		( (b3World*)world )->metalLastNarrowPhaseResultCount = resultCount;
		( (b3World*)world )->metalLastNarrowPhaseManifoldTableCount = world->contacts.count;
		*resultsOut = completedResults;
		*resultCountOut = resultCount;
		if ( residentBypassCountOut != NULL ) *residentBypassCountOut = residentBypassCount;
		context->contactTransitionCount = privateColdTopology ? 0 : transitionCount;
		if ( transitionsOut != NULL ) *transitionsOut = privateColdTopology == false && transitionCount > 0 ? completedTransitions : NULL;
		if ( transitionCountOut != NULL ) *transitionCountOut = privateColdTopology ? 0 : transitionCount;
		if ( stats != NULL )
		{
			stats->contactInputBootstrapDispatchCount = bootstrapInputs ? 1 : 0;
			stats->contactInputBootstrapSharedBytes = bootstrapInputs && pairSeedBootstrapInputs == false
				? (uint64_t)contactCount * sizeof( b3MetalContactInputSeed ) : 0;
			stats->contactInputBootstrapStatusSharedBytes = bootstrapInputs
				? sizeof( uint32_t ) : 0;
			stats->contactInputPrivateBytes = context->convexManifoldInputsPrivate
				? (uint64_t)contactCount * sizeof( b3MetalConvexManifoldInput ) : 0;
			stats->contactTransitionCount = privateColdTopology ? 0 : transitionCount;
			stats->contactTransitionSharedBytes = privateColdTopology == false && transitionCount > 0
				? (uint64_t)world->contacts.count * sizeof( b3MetalContactTransition ) : 0;
			stats->contactExceptionSharedBytes = (uint64_t)resultCount * sizeof( b3MetalConvexManifoldResult );
			stats->contactPrivateTopologySchedulePrivateBytes = privateColdTopology ? privateScheduleBytes : 0;
			stats->contactTopologySummarySharedBytes = privateColdTopology ? sizeof( b3MetalManifoldSummary ) : 0;
			b3MetalFillStats( commandBuffer, stats, encodeMs, waitMs,
				4 + ( bootstrapInputs ? 1 : 0 ), 3 + ( bootstrapInputs ? 1 : 0 ), 1, traceSite, gpuMsOverride );
		}
		return true;
	}
}

bool b3MetalComputeConvexManifolds( b3MetalContext* context, const b3World* world, const int* contactIndices,
	int contactCount, const b3MetalConvexManifoldResult** resultsOut, int* resultCountOut, int* residentBypassCountOut,
	const b3MetalContactTransition** transitionsOut, int* transitionCountOut,
	b3MetalDispatchStats* stats )
{
	if ( resultsOut != NULL ) *resultsOut = NULL;
	if ( resultCountOut != NULL ) *resultCountOut = 0;
	if ( residentBypassCountOut != NULL ) *residentBypassCountOut = 0;
	if ( transitionsOut != NULL ) *transitionsOut = NULL;
	if ( transitionCountOut != NULL ) *transitionCountOut = 0;
	if ( stats != NULL ) *stats = (b3MetalDispatchStats){ .bodyCount = contactCount };
	if ( context == NULL || world == NULL || contactCount < 0 || resultsOut == NULL || resultCountOut == NULL ||
		 ( transitionsOut == NULL ) != ( transitionCountOut == NULL ) )
	{
		return false;
	}
	b3MetalNarrowEncode encode = { 0 };
	id<MTLCommandBuffer> commandBuffer = nil;
	if ( b3MetalEncodeConvexManifolds( context, world, contactIndices, contactCount, residentBypassCountOut != NULL,
			 &encode, &commandBuffer ) == false )
	{
		return false;
	}
	if ( commandBuffer == nil )
	{
		// Empty-input early-out: encode already zeroed the table count.
		return true;
	}
	double preWaitMs = b3MetalMonotonicMs();
	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];
	double postWaitMs = b3MetalMonotonicMs();
	bool consumed = b3MetalConsumeConvexManifolds( context, world, commandBuffer, &encode, encode.encodeMs,
		postWaitMs - preWaitMs, -1.0, resultsOut, resultCountOut, residentBypassCountOut, transitionsOut,
		transitionCountOut, stats, "narrow_phase" );
	[commandBuffer release];
	return consumed;
}

// Phase-1 deferred narrow: encode the narrow phase now, stash the uncommitted
// buffer in the context, and return. The solve phase encodes into the same
// buffer (single commit/wait) via b3MetalSolveMergedSubsteps. Only the
// revision-stable reuse path is deferrable: cold bootstraps carry one-shot
// CPU validation that must run before any later dispatch, and private-cold
// topology takes its own solver route. Returns false when not deferrable or
// on any encode failure (caller uses the legacy path); a nil buffer with a
// true return means empty input (also legacy).
void b3MetalCancelPendingNarrow( b3MetalContext* context );
bool b3MetalDeferConvexManifolds( b3MetalContext* context, const b3World* world, int contactCount )
{
	if ( context == NULL || world == NULL || contactCount <= 0 || context->pendingNarrow )
	{
		return false;
	}
	if ( b3MetalCanReuseConvexManifoldInputs( context, world, contactCount ) == false )
	{
		return false;
	}
	b3MetalNarrowEncode encode = { 0 };
	id<MTLCommandBuffer> commandBuffer = nil;
	if ( b3MetalEncodeConvexManifolds( context, world, NULL, contactCount, true, &encode, &commandBuffer ) == false ||
		 commandBuffer == nil )
	{
		b3MetalCancelPendingNarrow( context );
		return false;
	}
	if ( encode.bootstrapInputs || encode.privateColdTopology )
	{
		// Encode consumed one-shot bootstrap authority or admitted a private
		// schedule; both need their legacy CPU handshake first. This cannot
		// happen under the reuse gate above, but fail closed regardless.
		// Balance Encode's retain of the uncommitted buffer being dropped.
		b3MetalCancelPendingNarrow( context );
		[commandBuffer release];
		return false;
	}
	context->pendingNarrowBuffer = commandBuffer;
	context->pendingNarrowEncode = encode;
	context->pendingNarrow = true;
	return true;
}

void b3MetalCancelPendingNarrow( b3MetalContext* context )
{
	if ( context == NULL )
	{
		return;
	}
	[context->pendingNarrowBuffer release];
	context->pendingNarrowBuffer = nil;
	context->pendingNarrowEncode = (b3MetalNarrowEncode){ 0 };
	context->pendingNarrow = false;
	context->mergeConsume = false;
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

static bool b3MetalApplyContactManifoldResult( b3World* world, b3Contact* contact,
	const b3MetalConvexManifoldResult* result )
{
	if ( world == NULL || contact == NULL || result == NULL || contact->contactId < 0 || contact->manifoldCount != 1 ||
		result->eligible == 0 || result->touching == 0 || result->pointCount < 1 ||
		result->pointCount > B3_MAX_MANIFOLD_POINTS || result->contactId != (uint32_t)contact->contactId ||
		result->inputIndex != (uint32_t)contact->contactId || result->contactGeneration != contact->generation )
	{
		return false;
	}

	if ( contact->manifolds == NULL )
	{
		// Cold resident contacts deliberately have no CPU manifold allocation.
		// Materialize exactly one only at this explicit CPU readback boundary.
		contact->manifolds = b3AllocateManifoldsSerial( world, 1 );
	}
	b3Manifold* manifold = contact->manifolds;
	manifold->normal = (b3Vec3){ result->normalX, result->normalY, result->normalZ };
	manifold->pointCount = (int)result->pointCount;
	const b3Vec3 anchorAs[B3_MAX_MANIFOLD_POINTS] = {
		{ result->point1X, result->point1Y, result->point1Z },
		{ result->point2X, result->point2Y, result->point2Z },
		{ result->point3X, result->point3Y, result->point3Z },
		{ result->point4X, result->point4Y, result->point4Z },
	};
	const b3Vec3 anchorBs[B3_MAX_MANIFOLD_POINTS] = {
		{ result->anchorB1X, result->anchorB1Y, result->anchorB1Z },
		{ result->anchorB2X, result->anchorB2Y, result->anchorB2Z },
		{ result->anchorB3X, result->anchorB3Y, result->anchorB3Z },
		{ result->anchorB4X, result->anchorB4Y, result->anchorB4Z },
	};
	const float separations[B3_MAX_MANIFOLD_POINTS] = {
		result->separation1, result->separation2, result->separation3, result->separation4,
	};
	const float normalImpulses[B3_MAX_MANIFOLD_POINTS] = {
		result->normalImpulse1, result->normalImpulse2, result->normalImpulse3, result->normalImpulse4,
	};
	const uint32_t featureIds[B3_MAX_MANIFOLD_POINTS] = {
		result->featureId1, result->featureId2, result->featureId3, result->featureId4,
	};
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
	uint8_t satType = (uint8_t)result->satCache;
	if ( satType == b3_faceAxisA || satType == b3_faceAxisB || satType == b3_edgePairAxis )
	{
		contact->convexContact.cache.satCache = (b3SATCache){
			.separation = result->satSeparation,
			.type = satType,
			.indexA = (uint8_t)( result->satCache >> 8 ),
			.indexB = (uint8_t)( result->satCache >> 16 ),
			.hit = (uint8_t)( result->satCache >> 24 ),
		};
	}
	contact->flags &= ~b3_simMetalManifoldStale;
	return true;
}

bool b3MetalSyncContactManifold( b3MetalContext* context, b3World* world, b3Contact* contact )
{
	if ( context == NULL || world == NULL || contact == NULL ) return false;
	if ( contact->contactId < 0 || contact->contactId >= context->convexManifoldTableCount ) return false;
	NSUInteger offset = (NSUInteger)contact->contactId * sizeof( b3MetalConvexManifoldResult );
	if ( b3MetalReadbackConvexManifoldRange( context, offset, sizeof( b3MetalConvexManifoldResult ) ) == false ) return false;
	return b3MetalApplyContactManifoldResult( world, contact, context->convexManifoldTableReadbackBuffer.contents );
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
			b3MetalApplyContactManifoldResult( world, contact, results + contactId ) == false ) return false;
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
	b3MetalPairSummary* summary = context->shapeSummaryBuffer.contents;
	if ( summary != NULL && summary->flags == 0 && summary->totalCount <= (uint64_t)context->shapeBoundsCount &&
		 summary->totalCount <= INT32_MAX )
	{
		context->residentPairMoveCount = (int)summary->totalCount;
		context->residentPairMovesValid = true;
	}
	else
	{
		context->residentPairMoveCount = 0;
		context->residentPairMovesValid = false;
	}
}

int b3MetalGetResidentPairMoveCount( const b3MetalContext* context )
{
	return context != NULL && context->residentPairMovesValid ? context->residentPairMoveCount : 0;
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
	b3BroadPhase* broadPhase = &world->broadPhase;
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
		if ( shape->proxyKey != B3_NULL_INDEX && B3_PROXY_TYPE( shape->proxyKey ) != b3_staticBody )
		{
			// The CPU tree may have missed several resident enlargements even when
			// the latest result remained inside its prior device fat bound. Restore
			// every moving leaf conservatively and buffer it for a complete CPU
			// fallback. This is an explicit observation/mutation boundary, not the
			// steady resident route.
			b3BodyType proxyType = B3_PROXY_TYPE( shape->proxyKey );
			int proxyId = B3_PROXY_ID( shape->proxyKey );
			b3AABB current = broadPhase->trees[proxyType].nodes[proxyId].aabb;
			b3AABB restored = {
				{ b3MinFloat( current.lowerBound.x, shape->fatAABB.lowerBound.x ),
				  b3MinFloat( current.lowerBound.y, shape->fatAABB.lowerBound.y ),
				  b3MinFloat( current.lowerBound.z, shape->fatAABB.lowerBound.z ) },
				{ b3MaxFloat( current.upperBound.x, shape->fatAABB.upperBound.x ),
				  b3MaxFloat( current.upperBound.y, shape->fatAABB.upperBound.y ),
				  b3MaxFloat( current.upperBound.z, shape->fatAABB.upperBound.z ) },
			};
			shape->fatAABB = restored;
			if ( memcmp( &current, &restored, sizeof( current ) ) != 0 )
			{
				b3BroadPhase_EnlargeProxy( broadPhase, shape->proxyKey, restored );
			}
			else
			{
				b3BufferMove( broadPhase, shape->proxyKey );
			}
			shape->flags &= ~b3_enlargedAABB;
		}
	}
	world->metalShapeCpuBoundsStale = false;
	context->residentPairMoveCount = 0;
	context->residentPairMovesValid = false;
	// CPU ancestor bounds are conservative but need not be byte-identical to
	// the complete device refit, so force the next Metal query to re-upload.
	context->pairTreeRevision = 0;
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
		bool privateColdSchedule = context->solvingPrivateColdSchedule;
		int activeColorCount = privateColdSchedule ? 1 : stepContext->activeColorCount;
		int wideContactCount = stepContext->wideContactCount;
		NSUInteger stateBytes = (NSUInteger)bodyCount * sizeof( b3BodyState );
		NSUInteger propertyBytes = (NSUInteger)bodyCount * sizeof( b3MetalBodyProperties );
		NSUInteger contactBytes = (NSUInteger)wideContactCount * sizeof( b3ContactConstraintWide );
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
		for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
		{
			distanceOffsets[colorIndex] = coloredDistanceCount;
			parallelOffsets[colorIndex] = coloredParallelCount;
			int count = privateColdSchedule ? 0 : stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = privateColdSchedule ? NULL : stepContext->jointPrepareSpans[colorIndex].joints;
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
		bool reuseFinalizeProperties = publishBodyTransforms && context->finalizePropertiesResidentCount == bodyCount &&
			context->finalizePropertiesResidentRevision == stepContext->world->metalBodyPropertyRevision;
		// A failed command must not leave the previous generation reusable.
		context->bodyStateResidentCount = 0;
		context->bodyPropertiesResidentCount = 0;
		context->finalizePropertiesResidentCount = 0;
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
		if ( finalizeBodies && reuseFinalizeProperties == false )
		{
			b3MetalPackFinalizeProperties( context->finalizePropertiesBuffer.contents, stepContext->sims, bodyCount,
				stepContext->world );
		}
		int shapeCount = finalizeBodies ? b3MetalPackShapeInputs( context, stepContext ) : 0;
		bool constraintsAlreadyShared = privateColdSchedule || contactBytes == 0 ||
			stepContext->wideConstraints == context->contactConstraintBuffer.contents;
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
		for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
		{
			int count = privateColdSchedule ? 0 : stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = privateColdSchedule ? NULL : stepContext->jointPrepareSpans[colorIndex].joints;
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

		// Phase-1 merge: when a deferred narrow is pending, encode into the same
		// command buffer (its narrow encoder already ended; cross-encoder
		// ordering is implicit) so one commit/wait covers both phases. The
		// buffer stays owned by the context (see Defer/Cancel/destroy); the
		// merged wrapper releases it after this call returns.
		bool mergeBorrowedBuffer = context->mergeConsume && context->pendingNarrowBuffer != nil;
		if ( context->mergeConsume == false && context->pendingNarrow )
		{
			B3_ASSERT( 0 && "legacy solve with a pending narrow" );
			b3MetalCancelPendingNarrow( context );
		}
		id<MTLCommandBuffer> commandBuffer =
			mergeBorrowedBuffer ? context->pendingNarrowBuffer : [context->queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
		if ( commandBuffer == nil || encoder == nil )
		{
			return false;
		}

		double encodeStartMs = b3MetalMonotonicMs();

		id<MTLBuffer> contactScheduleBuffer = privateColdSchedule
			? context->privateColdContactScheduleBuffer : context->contactPrepareIndexBuffer;
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
				.wideCount = (uint32_t)wideContactCount,
				.tableCount = (uint32_t)context->convexManifoldTableCount,
				.warmStartScale = stepContext->enableWarmStarting ? 1.0f : 0.0f,
				.invTau = 1.0f / B3_SPECULATIVE_DISTANCE,
				.contactSoftness = stepContext->contactSoftness,
				.staticSoftness = stepContext->staticSoftness,
				.generation = context->contactPrepareGeneration,
			};
			[encoder setComputePipelineState:context->prepareContactsPipeline];
			[encoder setBuffer:contactScheduleBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->convexManifoldTableBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->bodyPropertiesBuffer offset:0 atIndex:3];
			[encoder setBuffer:context->bodyStateBuffer offset:0 atIndex:4];
			[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:5];
			[encoder setBuffer:context->contactPrepareStatusBuffer offset:0 atIndex:6];
			[encoder setBytes:&params length:sizeof( params ) atIndex:7];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideContactCount, 1, 1 )
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
				for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
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
					int wideOffset = privateColdSchedule ? 0 : stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = privateColdSchedule ? wideContactCount :
						stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
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
					int meshOffset = privateColdSchedule ? 0 : stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = privateColdSchedule ? 0 :
						stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
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
				for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
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
					int wideOffset = privateColdSchedule ? 0 : stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = privateColdSchedule ? wideContactCount :
						stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
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
					int meshOffset = privateColdSchedule ? 0 : stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = privateColdSchedule ? 0 :
						stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
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
				for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
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
					int wideOffset = privateColdSchedule ? 0 : stepContext->widePrepareSpans[colorIndex].start;
					int wideCount = privateColdSchedule ? wideContactCount :
						stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
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
					int meshOffset = privateColdSchedule ? 0 : stepContext->contactPrepareSpans[colorIndex].start;
					int meshCount = privateColdSchedule ? 0 :
						stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
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
			b3MetalHasRestitution( stepContext->wideConstraints, wideContactCount );
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
			for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
			{
				bool dispatched = false;
				int wideOffset = privateColdSchedule ? 0 : stepContext->widePrepareSpans[colorIndex].start;
				int wideCount = privateColdSchedule ? wideContactCount :
					stepContext->widePrepareSpans[colorIndex + 1].start - wideOffset;
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
				int meshOffset = privateColdSchedule ? 0 : stepContext->contactPrepareSpans[colorIndex].start;
				int meshCount = privateColdSchedule ? 0 :
					stepContext->contactPrepareSpans[colorIndex + 1].start - meshOffset;
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
				.wideCount = (uint32_t)wideContactCount,
				.tableCount = (uint32_t)context->convexManifoldTableCount,
				.generation = context->contactPrepareGeneration,
			};
			[encoder setComputePipelineState:context->storeContactImpulsesPipeline];
			[encoder setBuffer:contactScheduleBuffer offset:0 atIndex:0];
			[encoder setBuffer:context->contactConstraintBuffer offset:0 atIndex:1];
			[encoder setBuffer:context->contactImpulseResultBuffer offset:0 atIndex:2];
			[encoder setBuffer:context->contactPrepareTableBuffer offset:0 atIndex:3];
			[encoder setBytes:&params length:sizeof( params ) atIndex:4];
			[encoder dispatchThreads:MTLSizeMake( (NSUInteger)wideContactCount, 1, 1 )
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
		double preWaitMs = b3MetalMonotonicMs();
		[commandBuffer commit];
		[commandBuffer waitUntilCompleted];
		double postWaitMs = b3MetalMonotonicMs();
		if ( commandBuffer.status != MTLCommandBufferStatusCompleted )
		{
			// A merged buffer may have partially executed: drop impulse
			// authority before any legacy retry. Legacy path is unaffected
			// (nothing pending).
			if ( context->mergeConsume ) b3MetalInvalidateContactImpulseResults( context );
			b3MetalCancelPendingNarrow( context );
			return false;
		}
		bool mergedNarrow = context->mergeConsume;
		context->mergeConsume = false;
		context->mergeNarrowOk = false;
		context->mergeMispredict = false;
		double narrowGpuMs = 0.0;
		double mergedGpuMs = -1.0;
		if ( mergedNarrow )
		{
			// Single wait covered both phases. Consume the narrow results
			// first: the solve post below must not commit residency when the
			// stability prediction failed. GPU time is split by dispatch
			// share (documented approximation until per-stage timestamps).
			int narrowDispatches = 4;
			int solveDispatches = 10 + 13 * activeColorCount;
			mergedGpuMs = -1.0;
			if ( commandBuffer.GPUEndTime >= commandBuffer.GPUStartTime )
			{
				mergedGpuMs = 1000.0 * ( commandBuffer.GPUEndTime - commandBuffer.GPUStartTime );
			}
			narrowGpuMs = mergedGpuMs * (double)narrowDispatches / (double)( narrowDispatches + solveDispatches );
			const b3MetalConvexManifoldResult* narrowResults = NULL;
			int narrowResultCount = 0, narrowBypassCount = 0;
			const b3MetalContactTransition* narrowTransitions = NULL;
			int narrowTransitionCount = 0;
			b3MetalDispatchStats narrowStats = { 0 };
			if ( b3MetalConsumeConvexManifolds( context, stepContext->world, commandBuffer,
					 &context->pendingNarrowEncode, context->pendingNarrowEncode.encodeMs, 0.0, narrowGpuMs,
					 &narrowResults, &narrowResultCount, &narrowBypassCount, &narrowTransitions,
					 &narrowTransitionCount, &narrowStats, "narrow_merged" ) == false )
			{
				b3MetalInvalidateContactImpulseResults( context );
				b3MetalCancelPendingNarrow( context );
				return false;
			}
			context->mergeResults = narrowResults;
			context->mergeResultCount = narrowResultCount;
			context->mergeBypassCount = narrowBypassCount;
			context->mergeTransitions = narrowTransitions;
			context->mergeTransitionCount = narrowTransitionCount;
			context->mergeNarrowStats = narrowStats;
			context->mergeNarrowOk = true;
			if ( narrowResultCount != 0 || narrowTransitionCount != 0 )
			{
				// Stability prediction failed: exceptions or first-touch
				// transitions need the CPU middle. Discard the speculative
				// solve output (impulse table) without committing residency;
				// the caller runs the legacy middle and re-solves.
				b3MetalInvalidateContactImpulseResults( context );
				context->mergeMispredict = true;
				return false;
			}
		}
		if ( prepareContactsOnGpu && *(const uint32_t*)context->contactPrepareStatusBuffer.contents != 0 )
		{
			// Narrow outputs are usable (see above) but the solve half is
			// not: route recovery through the mispredict path so the legacy
			// solve is attempted before CPU fallback. Invalidate first: the
			// merged buffer executed with a flagged preparation.
			if ( mergedNarrow && context->mergeNarrowOk ) b3MetalInvalidateContactImpulseResults( context );
			return false;
		}
		if ( publishBodyTransforms )
		{
			b3MetalCommitBodyTransformDeviceRefresh( context, stepContext->world, bodyCount );
			context->bodyStateResidentRevision = stepContext->world->metalBodyStateRevision;
			context->bodyPropertiesResidentCount = bodyCount;
			context->bodyPropertiesResidentRevision = stepContext->world->metalBodyPropertyRevision;
			context->finalizePropertiesResidentCount = bodyCount;
			context->finalizePropertiesResidentRevision = stepContext->world->metalBodyPropertyRevision;
			context->bodyMoveResultCount = bodyCount;
			context->bodyMoveResultStepIndex = stepContext->world->stepIndex;
			stepContext->metalBodyStatesFinalizedOnDevice = true;
			stepContext->metalBodyMoveEventsOnDevice = true;
		}
		else
		{
			context->bodyStateResidentCount = 0;
			context->bodyPropertiesResidentCount = 0;
			context->finalizePropertiesResidentCount = 0;
		}
		if ( prepareContactsOnGpu )
		{
			context->contactImpulseResultCount = context->convexManifoldTableCount;
			context->contactImpulseResultGeneration = context->contactPrepareGeneration;
			stepContext->world->metalLastContactImpulseResultBytes =
				(uint64_t)stepContext->world->metalLastResidentConvexContactCount * sizeof( b3MetalContactImpulseResult );
		}

		bool keepBodyStatesResident = publishBodyTransforms && stepContext->metalFullyResidentConvexContacts &&
			prepareContactsOnGpu && stepContext->contactConstraintCount == 0 &&
			stepContext->overflowContactConstraintCount == 0 && stepContext->jointConstraintCount == 0 &&
			stepContext->overflowJointConstraintCount == 0;
		if ( keepBodyStatesResident )
		{
			stepContext->states = context->bodyStateBuffer.contents;
			b3AtomicStoreInt( &stepContext->world->metalBodyStateCpuStale, 1 );
			stepContext->world->metalLastBodyStateReadbackBytes = 0;
		}
		else
		{
			memcpy( stepContext->states, context->bodyStateBuffer.contents, stateBytes );
			b3AtomicStoreInt( &stepContext->world->metalBodyStateCpuStale, 0 );
			stepContext->world->metalLastBodyStateReadbackBytes = stateBytes;
		}
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
					stepContext->metalEnlargedShapeResults = NULL;
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
		for ( int colorIndex = 0; colorIndex < activeColorCount; ++colorIndex )
		{
			int count = privateColdSchedule ? 0 : stepContext->jointPrepareSpans[colorIndex + 1].start -
				stepContext->jointPrepareSpans[colorIndex].start;
			b3JointSim* joints = privateColdSchedule ? NULL : stepContext->jointPrepareSpans[colorIndex].joints;
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
		if ( stats != NULL )
		{
			// Analytic schedule count: fixed stages plus per-color solver
			// phases. Per-encode measurement replaces this formula in a later
			// phase; the timeline test locks it until then.
			int solveDispatches = 10 + 13 * activeColorCount;
			// Merged path: report the full single wait here (narrow reported
			// zero) and only the solve share of GPU time (narrow took its
			// share above). Legacy passes -1 to measure the buffer as before.
			double solveGpuOverride = mergedNarrow && mergedGpuMs >= 0.0 ? mergedGpuMs - narrowGpuMs : -1.0;
			b3MetalFillStats( commandBuffer, stats, preWaitMs - encodeStartMs, postWaitMs - preWaitMs,
				solveDispatches, solveDispatches, 1, mergedNarrow ? "solve_merged" : "solve", solveGpuOverride );
		}
		return true;
	}
}

// Phase-1 merged solve: encode the contact solve into the deferred narrow
// buffer (single commit/wait for both phases), then consume the narrow
// results before committing residency. Returns 1 on accept (speculative
// solve results valid), -1 on stability mispredict (narrow outputs in the
// context merge fields are valid; caller runs the legacy middle and
// re-solves), 0 on hard failure (caller falls back to CPU from a fresh
// legacy narrow). Narrow/solve stats are reported separately in the two out
// params (NULL allowed); the single wait is attributed to the solve stats.
int b3MetalSolveMergedSubsteps( b3MetalContext* context, b3StepContext* stepContext, int velocityIterations,
	int relaxIterations, int restitutionIterations, b3MetalDispatchStats* narrowStatsOut,
	b3MetalDispatchStats* solveStatsOut, int* mergedBypassCountOut )
{
	if ( context == NULL || stepContext == NULL || context->pendingNarrowBuffer == nil || context->pendingNarrow == false )
	{
		return 0;
	}
	if ( narrowStatsOut != NULL ) *narrowStatsOut = (b3MetalDispatchStats){ 0 };
	if ( solveStatsOut != NULL ) *solveStatsOut = (b3MetalDispatchStats){ 0 };
	if ( mergedBypassCountOut != NULL ) *mergedBypassCountOut = 0;
	context->mergeConsume = true;
	context->mergeNarrowOk = false;
	context->mergeMispredict = false;
	bool ok = b3MetalSolveContactSubsteps( context, stepContext, velocityIterations, relaxIterations,
		restitutionIterations, solveStatsOut );
	context->mergeConsume = false;
	// Balance Defer's retain: the borrowed buffer is fully consumed (merge
	// outputs live in context buffers, not the command buffer).
	[context->pendingNarrowBuffer release];
	context->pendingNarrowBuffer = nil;
	context->pendingNarrow = false;
	if ( ok )
	{
		if ( narrowStatsOut != NULL ) *narrowStatsOut = context->mergeNarrowStats;
		if ( mergedBypassCountOut != NULL ) *mergedBypassCountOut = context->mergeBypassCount;
		// One physical buffer covered both phases: count it once (solve
		// reports it below).
		if ( narrowStatsOut != NULL ) narrowStatsOut->commandBufferCount = 0;
		return 1;
	}
	if ( context->mergeNarrowOk )
	{
		if ( narrowStatsOut != NULL ) *narrowStatsOut = context->mergeNarrowStats;
		// Narrow valid; solve half unusable. mergeMispredict distinguishes a
		// stability miss (true: run legacy middle + re-solve) from a solve
		// failure with usable narrow outputs (false: same recovery; the
		// legacy solve re-validates from scratch).
		return -1;
	}
	b3MetalCancelPendingNarrow( context );
	return 0;
}

bool b3MetalFetchMergedNarrow( const b3MetalContext* context, const b3MetalConvexManifoldResult** resultsOut,
	int* resultCountOut, int* bypassCountOut, const b3MetalContactTransition** transitionsOut,
	int* transitionCountOut, b3MetalDispatchStats* statsOut )
{
	if ( resultsOut != NULL ) *resultsOut = NULL;
	if ( resultCountOut != NULL ) *resultCountOut = 0;
	if ( bypassCountOut != NULL ) *bypassCountOut = 0;
	if ( transitionsOut != NULL ) *transitionsOut = NULL;
	if ( transitionCountOut != NULL ) *transitionCountOut = 0;
	if ( statsOut != NULL ) *statsOut = (b3MetalDispatchStats){ 0 };
	if ( context == NULL || context->mergeNarrowOk == false ) return false;
	if ( resultsOut != NULL ) *resultsOut = context->mergeResults;
	if ( resultCountOut != NULL ) *resultCountOut = context->mergeResultCount;
	if ( bypassCountOut != NULL ) *bypassCountOut = context->mergeBypassCount;
	if ( transitionsOut != NULL ) *transitionsOut = context->mergeTransitions;
	if ( transitionCountOut != NULL ) *transitionCountOut = context->mergeTransitionCount;
	if ( statsOut != NULL ) *statsOut = context->mergeNarrowStats;
	return true;
}

bool b3MetalSolvePrivateColdContactSubsteps( b3MetalContext* context, b3StepContext* stepContext,
	const b3BodySim* bodySims, b3BodyState* bodyStates, int awakeBodyCount,
	int velocityIterations, int relaxIterations, int restitutionIterations, b3MetalDispatchStats* stats )
{
	int contactCount = 0, wideCount = 0;
	if ( context == NULL || stepContext == NULL || bodySims == NULL || bodyStates == NULL || awakeBodyCount <= 0 ||
		b3MetalHasPrivateColdContactSchedule( context, stepContext->world, &contactCount, &wideCount ) == false ||
		contactCount <= 0 || wideCount <= 0 || stepContext->wideContactCount != wideCount ||
		stepContext->metalResidentConvexContactCount != contactCount || stepContext->activeColorCount != 1 ||
		stepContext->contactConstraintCount != 0 || stepContext->overflowContactConstraintCount != 0 ||
		stepContext->jointConstraintCount != 0 || stepContext->overflowJointConstraintCount != 0 ||
		stepContext->world->solverSets.data[b3_awakeSet].bodyStates.count != awakeBodyCount )
	{
		return false;
	}
	b3BodySim* oldSims = stepContext->sims;
	b3BodyState* oldStates = stepContext->states;
	stepContext->sims = (b3BodySim*)bodySims;
	stepContext->states = bodyStates;
	context->solvingPrivateColdSchedule = true;
	bool success = b3MetalSolveContactSubsteps( context, stepContext, velocityIterations, relaxIterations,
		restitutionIterations, stats );
	context->solvingPrivateColdSchedule = false;
	if ( success == false )
	{
		stepContext->sims = oldSims;
		stepContext->states = oldStates;
		b3MetalCancelPrivateColdContactSchedule( context );
	}
	return success;
}
