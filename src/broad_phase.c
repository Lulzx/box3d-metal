// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "broad_phase.h"

#include "aabb.h"
#include "arena_allocator.h"
#include "body.h"
#include "contact.h"
#include "core.h"
#include "parallel_for.h"
#include "physics_world.h"
#include "platform.h"
#include "shape.h"
#if defined( BOX3D_METAL )
#include "metal_backend.h"
#endif

#include <string.h>

static void b3IncrementTreeRevision( b3BroadPhase* bp )
{
	bp->treeRevision += 1;
	if ( bp->treeRevision == 0 )
	{
		bp->treeRevision = 1;
	}
}

void b3CreateBroadPhase( b3BroadPhase* bp, const b3Capacity* capacity )
{
	_Static_assert( b3_bodyTypeCount == 3, "must be three body types" );

	bp->movedProxies[b3_staticBody] = b3CreateBitSet( b3MaxInt( 16, capacity->staticShapeCount ) );
	bp->movedProxies[b3_kinematicBody] = b3CreateBitSet( 16 );
	bp->movedProxies[b3_dynamicBody] = b3CreateBitSet( b3MaxInt( 16, capacity->dynamicShapeCount ) );
	b3Array_Reserve( bp->moveArray, capacity->dynamicShapeCount );
	bp->moveResults = NULL;
	bp->movePairs = NULL;
	bp->movePairCapacity = 0;
	b3AtomicStoreInt( &bp->movePairIndex, 0 );
	bp->pairSet = b3CreateSet( 2 * capacity->contactCount );
	bp->treeRevision = 1;

	int staticCapacity = b3MaxInt( 16, capacity->staticShapeCount );
	bp->trees[b3_staticBody] = b3DynamicTree_Create( staticCapacity );

	int kinematicCapacity = 16;
	bp->trees[b3_kinematicBody] = b3DynamicTree_Create( kinematicCapacity );

	int dynamicCapacity = b3MaxInt( 16, capacity->dynamicShapeCount );
	bp->trees[b3_dynamicBody] = b3DynamicTree_Create( dynamicCapacity );
}

void b3DestroyBroadPhase( b3BroadPhase* bp )
{
	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DynamicTree_Destroy( bp->trees + i );
	}

	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DestroyBitSet( &bp->movedProxies[i] );
	}
	b3Array_Destroy( bp->moveArray );
	b3DestroySet( &bp->pairSet );

	*bp = (b3BroadPhase){ 0 };

	memset( bp, 0, sizeof( b3BroadPhase ) );
}

static void b3UnBufferMove( b3BroadPhase* bp, int proxyKey )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );
	b3BitSet* set = &bp->movedProxies[proxyType];

	if ( b3GetBit( set, proxyId ) )
	{
		b3ClearBit( set, proxyId );

		// Purge from move buffer. Linear search.
		// todo if I can iterate the move set then I don't need the moveArray
		int count = bp->moveArray.count;
		for ( int i = 0; i < count; ++i )
		{
			if ( bp->moveArray.data[i] == proxyKey )
			{
				b3Array_RemoveSwap( bp->moveArray, i );
				break;
			}
		}
	}
}

int b3BroadPhase_CreateProxy( b3BroadPhase* bp, b3BodyType proxyType, b3AABB aabb, uint64_t categoryBits, int shapeIndex,
							  bool forcePairCreation )
{
	B3_ASSERT( 0 <= proxyType && proxyType < b3_bodyTypeCount );
	int proxyId = b3DynamicTree_CreateProxy( bp->trees + proxyType, aabb, categoryBits, shapeIndex );
	b3IncrementTreeRevision( bp );
	int proxyKey = B3_PROXY_KEY( proxyId, proxyType );
	if ( proxyType != b3_staticBody || forcePairCreation )
	{
		b3BufferMove( bp, proxyKey );
	}
	return proxyKey;
}

void b3BroadPhase_DestroyProxy( b3BroadPhase* bp, int proxyKey )
{
	b3UnBufferMove( bp, proxyKey );

	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	B3_ASSERT( 0 <= proxyType && proxyType <= b3_bodyTypeCount );
	b3DynamicTree_DestroyProxy( bp->trees + proxyType, proxyId );
	b3IncrementTreeRevision( bp );
}

void b3BroadPhase_MoveProxy( b3BroadPhase* bp, int proxyKey, b3AABB aabb )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	b3DynamicTree_MoveProxy( bp->trees + proxyType, proxyId, aabb );
	b3IncrementTreeRevision( bp );
	b3BufferMove( bp, proxyKey );
}

void b3BroadPhase_EnlargeProxy( b3BroadPhase* bp, int proxyKey, b3AABB aabb )
{
	B3_ASSERT( proxyKey != B3_NULL_INDEX );
	int typeIndex = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	B3_ASSERT( typeIndex != b3_staticBody );

	b3DynamicTree_EnlargeProxy( bp->trees + typeIndex, proxyId, aabb );
	b3IncrementTreeRevision( bp );
	b3BufferMove( bp, proxyKey );
}

typedef struct b3MovePair
{
	int shapeIndexA;
	int shapeIndexB;
	int childIndex;
	b3MovePair* next;
	bool heap;
} b3MovePair;

typedef struct b3MoveResult
{
	b3MovePair* pairList;
} b3MoveResult;

typedef struct b3QueryPairContext
{
	b3World* world;
	b3MoveResult* moveResult;
	b3AABB aabb;
	b3BodyType queryTreeType;
	int queryProxyKey;
	int queryShapeIndex;

	int compoundProxyId;
	int compoundShapeIndex;
} b3QueryPairContext;

static void b3AppendMovePair( b3QueryPairContext* queryContext, b3Shape* shapeA, b3Shape* shapeB, int childIndex )
{
	b3World* world = queryContext->world;
	int bodyIdA = shapeA->bodyId;
	int bodyIdB = shapeB->bodyId;
	b3Body* bodyA = b3Array_Get( world->bodies, bodyIdA );
	b3Body* bodyB = b3Array_Get( world->bodies, bodyIdB );
	if ( b3ShouldBodiesCollide( world, bodyA, bodyB ) == false )
	{
		return;
	}

	if ( ( shapeA->flags & b3_enableCustomFiltering ) || ( shapeB->flags & b3_enableCustomFiltering ) )
	{
		b3CustomFilterFcn* customFilterFcn = world->customFilterFcn;
		if ( customFilterFcn != NULL )
		{
			b3ShapeId idA = { shapeA->id + 1, world->worldId, shapeA->generation };
			b3ShapeId idB = { shapeB->id + 1, world->worldId, shapeB->generation };
			if ( customFilterFcn( idA, idB, world->customFilterContext ) == false )
			{
				return;
			}
		}
	}

	b3BroadPhase* broadPhase = &world->broadPhase;
	int pairIndex = b3AtomicFetchAddInt( &broadPhase->movePairIndex, 1 );
	if ( pairIndex >= broadPhase->movePairCapacity )
	{
		return;
	}

	b3MovePair* pair = broadPhase->movePairs + pairIndex;
	pair->heap = false;
	pair->shapeIndexA = shapeA->id;
	pair->shapeIndexB = shapeB->id;
	pair->childIndex = childIndex;
	pair->next = queryContext->moveResult->pairList;
	queryContext->moveResult->pairList = pair;
}

// This is called from b3DynamicTree::Query when we are gathering pairs.
static bool b3PairQueryCallback( int proxyId, uint64_t userData, void* context )
{
	b3QueryPairContext* queryContext = (b3QueryPairContext*)context;
	b3World* world = queryContext->world;
	int shapeIndex;
	int childIndex = 0;

	if ( queryContext->compoundShapeIndex == B3_NULL_INDEX )
	{
		// Outer query: userData is a shape index.
		shapeIndex = (int)userData;

		// A proxy cannot form a pair with itself.
		if ( shapeIndex == queryContext->queryShapeIndex )
		{
			return true;
		}

		b3Shape* shape = b3Array_Get( world->shapes, shapeIndex );
		if ( shape->type == b3_compoundShape )
		{
			// Query bounds are float world space, so the demoted transform is the matching float frame
			b3Transform compoundTransform = b3ToRelativeTransform( b3GetBodyTransform( world, shape->bodyId ), b3Pos_zero );
			b3AABB localAABB = b3AABB_Transform( b3InvertTransform( compoundTransform ), queryContext->aabb );

			// recurse
			queryContext->compoundShapeIndex = shapeIndex;
			queryContext->compoundProxyId = proxyId;

			b3DynamicTree_Query( &shape->compound->tree, localAABB, B3_DEFAULT_MASK_BITS, false, b3PairQueryCallback, context );
			queryContext->compoundShapeIndex = B3_NULL_INDEX;
			queryContext->compoundProxyId = B3_NULL_INDEX;
			return true;
		}
	}
	else
	{
		// Inner query into a compound shape: userData is the compound child index, not a shape
		// index, so do not compare it against queryShapeIndex.
		shapeIndex = queryContext->compoundShapeIndex;
		proxyId = queryContext->compoundProxyId;
		childIndex = (int)userData;
	}

	b3BroadPhase* broadPhase = &queryContext->world->broadPhase;

	int proxyKey = B3_PROXY_KEY( proxyId, queryContext->queryTreeType );
	int queryProxyKey = queryContext->queryProxyKey;

	// A proxy cannot form a pair with itself.
	B3_ASSERT( proxyKey != queryContext->queryProxyKey );

	b3BodyType treeType = queryContext->queryTreeType;
	b3BodyType queryProxyType = B3_PROXY_TYPE( queryProxyKey );

	// De-duplication
	// It is important to prevent duplicate contacts from being created. Ideally I can prevent duplicates
	// early and in the worker. Most of the time the movedProxies bit sets contain dynamic and kinematic
	// proxies, but sometimes static proxies are in there too (b3ShapeDef::invokeContactCreation or a
	// modified static shape), so we always have to check.

	// Is this proxy also moving?
	if ( queryProxyType == b3_dynamicBody )
	{
		if ( treeType == b3_dynamicBody && proxyKey < queryProxyKey )
		{
			bool moved = b3GetBit( &broadPhase->movedProxies[treeType], proxyId );
			if ( moved )
			{
				// Both proxies are moving. Avoid duplicate pairs.
				return true;
			}
		}
	}
	else
	{
		B3_ASSERT( treeType == b3_dynamicBody );
		bool moved = b3GetBit( &broadPhase->movedProxies[treeType], proxyId );
		if ( moved )
		{
			// Both proxies are moving. Avoid duplicate pairs.
			return true;
		}
	}

	uint64_t pairKey = b3ShapePairKey( shapeIndex, queryContext->queryShapeIndex, childIndex );
	if ( b3ContainsKey( &broadPhase->pairSet, pairKey ) )
	{
		// contact exists
		return true;
	}

	// Order shapes so that B3_SHAPE_PAIR_KEY works correctly
	int shapeIdA = shapeIndex;
	int shapeIdB = queryContext->queryShapeIndex;
	b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
	b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );
	int bodyIdA = shapeA->bodyId;
	int bodyIdB = shapeB->bodyId;

	// Are the shapes on the same body?
	if ( bodyIdA == bodyIdB )
	{
		return true;
	}

	// Sensors are handled elsewhere
	if ( shapeA->sensorIndex != B3_NULL_INDEX || shapeB->sensorIndex != B3_NULL_INDEX )
	{
		return true;
	}

	if ( b3ShouldShapesCollide( shapeA->filter, shapeB->filter ) == false )
	{
		return true;
	}

	b3AppendMovePair( queryContext, shapeA, shapeB, childIndex );

	// continue the query
	return true;
}

static void b3FindPairsTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3TracyCZoneNC( pair_task, "Pair Task", b3_colorAquamarine, true );

	B3_UNUSED( workerIndex );

	b3World* world = (b3World*)context;
	b3BroadPhase* bp = &world->broadPhase;

	b3QueryPairContext queryContext = { 0 };
	queryContext.world = world;
	queryContext.compoundShapeIndex = B3_NULL_INDEX;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		// Initialize move result for this moved proxy
		queryContext.moveResult = bp->moveResults + i;
		queryContext.moveResult->pairList = NULL;

		int proxyKey = bp->moveArray.data[i];
		b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );

		int proxyId = B3_PROXY_ID( proxyKey );
		queryContext.queryProxyKey = proxyKey;

		const b3DynamicTree* baseTree = bp->trees + proxyType;

		// We have to query the tree with the fat AABB so that
		// we don't fail to create a contact that may touch later.
		b3AABB fatAABB = b3DynamicTree_GetAABB( baseTree, proxyId );
		queryContext.queryShapeIndex = (int)b3DynamicTree_GetUserData( baseTree, proxyId );
		queryContext.aabb = fatAABB;

		// Compound shape collision invocation is not supported
		B3_VALIDATE( world->shapes.data[queryContext.queryShapeIndex].type != b3_compoundShape );

		// Query trees. Only dynamic proxies collide with kinematic and static proxies.
		// Using B3_DEFAULT_MASK_BITS so that b3Filter::groupIndex works.
		// consider using bits = groupIndex > 0 ? B3_DEFAULT_MASK_BITS : maskBits
		bool requireAllBits = false;
		if ( proxyType == b3_dynamicBody )
		{
			queryContext.queryTreeType = b3_kinematicBody;
			b3DynamicTree_Query( bp->trees + b3_kinematicBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits, b3PairQueryCallback,
								 &queryContext );

			queryContext.queryTreeType = b3_staticBody;
			b3DynamicTree_Query( bp->trees + b3_staticBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits, b3PairQueryCallback,
								 &queryContext );
		}

		// All proxies collide with dynamic proxies
		// Using B3_DEFAULT_MASK_BITS so that b3Filter::groupIndex works.
		queryContext.queryTreeType = b3_dynamicBody;
		b3DynamicTree_Query( bp->trees + b3_dynamicBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits, b3PairQueryCallback,
							 &queryContext );
	}

	b3TracyCZoneEnd( pair_task );
}

#if defined( BOX3D_METAL )
typedef struct b3MetalFindPairsContext
{
	b3World* world;
	const b3MetalPairQueryRecord* records;
	const b3MetalPairCandidate* candidates;
	const int* cpuFilterMoves;
} b3MetalFindPairsContext;

// Implements b3ParallelForCallback. Metal has already traversed the trees and
// rejected exact moved-proxy duplicates, existing non-compound contacts, and
// built-in shape filters. Only GPU-compacted joint/custom/compound exception
// moves reach this task; ordinary ranges go directly to serial contact commit.
static void b3FindPairsMetalTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3TracyCZoneNC( pair_metal_consume, "Pair Metal Consume", b3_colorAquamarine, true );
	B3_UNUSED( workerIndex );
	b3MetalFindPairsContext* metalContext = context;
	b3World* world = metalContext->world;
	b3BroadPhase* bp = &world->broadPhase;

	b3QueryPairContext queryContext = { 0 };
	queryContext.world = world;
	queryContext.compoundShapeIndex = B3_NULL_INDEX;
	queryContext.compoundProxyId = B3_NULL_INDEX;
	for ( int filterIndex = startIndex; filterIndex < endIndex; ++filterIndex )
	{
		int moveIndex = metalContext->cpuFilterMoves != NULL ? metalContext->cpuFilterMoves[filterIndex] : filterIndex;
		queryContext.moveResult = bp->moveResults + filterIndex;
		queryContext.moveResult->pairList = NULL;

		const b3MetalPairQueryRecord* record = metalContext->records + moveIndex;
		int queryProxyKey = record->queryProxyKey;
		queryContext.queryProxyKey = queryProxyKey;
		queryContext.queryShapeIndex = record->queryShapeIndex;
		queryContext.aabb = (b3AABB){
			{ record->lowerX, record->lowerY, record->lowerZ },
			{ record->upperX, record->upperY, record->upperZ },
		};
		B3_VALIDATE( world->shapes.data[queryContext.queryShapeIndex].type != b3_compoundShape );

		for ( uint32_t candidateIndex = 0; candidateIndex < record->count; ++candidateIndex )
		{
			const b3MetalPairCandidate* candidate = metalContext->candidates + record->offset + candidateIndex;
			queryContext.queryTreeType = (b3BodyType)candidate->treeType;
			b3Shape* shapeA = b3Array_Get( world->shapes, candidate->shapeIndex );
			if ( shapeA->type == b3_compoundShape )
			{
				b3PairQueryCallback( candidate->proxyId, (uint64_t)(uint32_t)candidate->shapeIndex, &queryContext );
				continue;
			}

			b3Shape* shapeB = b3Array_Get( world->shapes, queryContext.queryShapeIndex );
			B3_ASSERT( shapeA->id != shapeB->id );
			B3_ASSERT( shapeA->bodyId != shapeB->bodyId );
			B3_ASSERT( shapeA->sensorIndex == B3_NULL_INDEX && shapeB->sensorIndex == B3_NULL_INDEX );
			B3_ASSERT( b3ShouldShapesCollide( shapeA->filter, shapeB->filter ) );
			B3_ASSERT( b3ContainsKey( &bp->pairSet, b3ShapePairKey( shapeA->id, shapeB->id, 0 ) ) == false );
			b3AppendMovePair( &queryContext, shapeA, shapeB, 0 );
		}
	}
	b3TracyCZoneEnd( pair_metal_consume );
}
#endif

void b3BroadPhase_RebuildTrees( b3World* world )
{
	b3TracyCZoneNC( tree_task, "Rebuild Trees", b3_colorFireBrick, true );

	int dynamicLeafCount = b3DynamicTree_Rebuild( world->broadPhase.trees + b3_dynamicBody, false );
	int kinematicLeafCount = b3DynamicTree_Rebuild( world->broadPhase.trees + b3_kinematicBody, false );
	if ( dynamicLeafCount > 1 || kinematicLeafCount > 1 )
	{
		b3IncrementTreeRevision( &world->broadPhase );
	}

	b3TracyCZoneEnd( tree_task );
}

static void b3UpdateTreesTask( void* context )
{
	b3BroadPhase_RebuildTrees( (b3World*)context );
}

void b3UpdateBroadPhasePairs( b3World* world )
{
	b3BroadPhase* bp = &world->broadPhase;

	int moveCount = bp->moveArray.count;
#if defined( BOX3D_METAL )
	int residentMoveCount = world->metalBroadPhaseEnabled ? b3MetalGetResidentPairMoveCount( world->metalContext ) : 0;
	if ( moveCount > 0 && residentMoveCount > 0 )
	{
		// A CPU mutation arrived while a private finalization list was pending.
		// Restore the CPU oracle and let its de-duplicated move array own this step.
		if ( b3MetalSyncAllShapeBounds( world->metalContext, world ) == false )
		{
			world->metalPairFallbackCount += 1;
			return;
		}
		moveCount = bp->moveArray.count;
		residentMoveCount = 0;
	}
	bool useResidentMoves = moveCount == 0 && residentMoveCount > 0;
	if ( useResidentMoves ) moveCount = residentMoveCount;
#endif

	if ( moveCount == 0 )
	{
		return;
	}

	b3TracyCZoneNC( update_pairs, "Pairs", b3_colorMediumSlateBlue, true );

	b3Stack* alloc = &world->stack;
	bp->moveResults = NULL;
	bp->movePairs = NULL;
	int minRange = 64;
	bool usedMetalPairs = false;
	bool pairsAreEmpty = false;
#if defined( BOX3D_METAL )
	bool directMetalPlan = false;
	const b3MetalPairQueryRecord* metalRecords = NULL;
	const b3MetalPairCandidate* metalCandidates = NULL;
	const int* metalCpuFilterMoves = NULL;
	int candidateCount = 0;
	int cpuFilterMoveCount = 0;
#endif
#ifndef NDEBUG
	extern b3AtomicInt b3_probeCount;
	b3AtomicStoreInt( &b3_probeCount, 0 );
#endif
#if defined( BOX3D_METAL )
	if ( world->metalBroadPhaseEnabled && ( useResidentMoves || moveCount >= world->metalMinimumBodyCount ) )
	{
		b3MetalDispatchStats stats = { 0 };
		const int* moveArray = useResidentMoves ? NULL : bp->moveArray.data;
		if ( b3MetalGeneratePairCandidates( world->metalContext, world, moveArray, moveCount, &metalRecords, &metalCandidates,
				&candidateCount, &metalCpuFilterMoves, &cpuFilterMoveCount, &stats ) )
		{
			pairsAreEmpty = candidateCount == 0;
			directMetalPlan = candidateCount <= 16 * moveCount;
			if ( pairsAreEmpty == false )
			{
				// Ordinary records are already the final pair plan. Preserve the
				// historical fixed-capacity behavior by using the legacy all-record
				// consume path if the GPU candidate count exceeds that bound.
				int filterCount = directMetalPlan ? cpuFilterMoveCount : moveCount;
				if ( filterCount > 0 )
				{
					bp->moveResults = (b3MoveResult*)b3StackAlloc( alloc, filterCount * sizeof( b3MoveResult ), "move results" );
					bp->movePairCapacity = 16 * moveCount;
					bp->movePairs = (b3MovePair*)b3StackAlloc( alloc, bp->movePairCapacity * sizeof( b3MovePair ), "move pairs" );
					b3AtomicStoreInt( &bp->movePairIndex, 0 );
					b3MetalFindPairsContext metalContext = {
						world,
						metalRecords,
						metalCandidates,
						directMetalPlan ? metalCpuFilterMoves : NULL,
					};
					b3ParallelFor( world, b3FindPairsMetalTask, filterCount, minRange, &metalContext, "pairs metal filter" );
				}
			}

			int cpuFilterCandidateCount = 0;
			int directCreateCount = 0;
			if ( directMetalPlan )
			{
				for ( int moveIndex = 0; moveIndex < moveCount; ++moveIndex )
				{
					int count = (int)metalRecords[moveIndex].count;
					if ( metalRecords[moveIndex].requiresCpuFiltering ) cpuFilterCandidateCount += count;
					else directCreateCount += count;
				}
				world->metalPairCpuCandidateTraversalBypassCount += directCreateCount > 0 ? 1 : 0;
			}
			world->metalPairDispatchCount += 1;
			world->metalResidentPairMoveDispatchCount += useResidentMoves ? 1 : 0;
			world->metalPairTreeUploadCount += (uint64_t)stats.treeUploadCount;
			world->metalPairMetadataUploadCount += (uint64_t)stats.metadataUploadCount;
			world->metalPairSetUploadCount += (uint64_t)stats.pairSetUploadCount;
			world->metalLastPairMoveCount = moveCount;
			world->metalLastPairCandidateCount = candidateCount;
			world->metalLastPairMoveUploadBytes = useResidentMoves ? 0 : (uint64_t)moveCount * sizeof( int );
			world->metalLastPairCpuFilterMoveCount = directMetalPlan ? cpuFilterMoveCount : moveCount;
			world->metalLastPairCpuFilterCandidateCount = directMetalPlan ? cpuFilterCandidateCount : candidateCount;
			world->metalLastPairDirectCreateCount = directMetalPlan ? directCreateCount : 0;
			world->metalLastPairGpuMilliseconds = stats.gpuMilliseconds;
			usedMetalPairs = true;
		}
		else
		{
			world->metalPairFallbackCount += 1;
			if ( useResidentMoves )
			{
				if ( b3MetalSyncAllShapeBounds( world->metalContext, world ) == false )
				{
					b3TracyCZoneEnd( update_pairs );
					return;
				}
				moveCount = bp->moveArray.count;
				useResidentMoves = false;
			}
		}
	}
#endif
	if ( usedMetalPairs == false )
	{
		bp->moveResults = (b3MoveResult*)b3StackAlloc( alloc, moveCount * sizeof( b3MoveResult ), "move results" );
		bp->movePairCapacity = 16 * moveCount;
		bp->movePairs = (b3MovePair*)b3StackAlloc( alloc, bp->movePairCapacity * sizeof( b3MovePair ), "move pairs" );
		b3AtomicStoreInt( &bp->movePairIndex, 0 );
		b3ParallelFor( world, b3FindPairsTask, moveCount, minRange, world, "pairs" );
	}

	b3TracyCZoneNC( create_contacts, "Create Contacts", b3_colorCoral, true );

	// Task that can be done in parallel with the narrow-phase
	// - rebuild the collision tree for dynamic and kinematic bodies to keep their query performance good
	if ( usedMetalPairs )
	{
		// Keep the topology used by the resident Metal snapshot stable. CPU
		// leaves remain conservative; Metal refits exact internal bounds.
		world->userTreeTask = NULL;
	}
	else if ( world->taskCount < B3_MAX_TASKS )
	{
		world->userTreeTask = world->enqueueTaskFcn( &b3UpdateTreesTask, world, world->userTaskContext, "rebuild tree" );
		world->taskCount += 1;
		world->activeTaskCount += world->userTreeTask == NULL ? 0 : 1;
	}
	else
	{
		world->userTreeTask = NULL;
		b3UpdateTreesTask( world );
	}

	// Single-threaded work
	// - Create contacts in deterministic order
	// GPU traversal emits candidates in callback order. Erin's CPU staging path
	// prepends them, so ordinary GPU-planned ranges are consumed in reverse.
	for ( int i = 0; pairsAreEmpty == false && i < moveCount; ++i )
	{
#if defined( BOX3D_METAL )
		if ( directMetalPlan && metalRecords[i].requiresCpuFiltering == 0 )
		{
			const b3MetalPairQueryRecord* record = metalRecords + i;
			b3Shape* shapeB = b3Array_Get( world->shapes, record->queryShapeIndex );
			for ( uint32_t candidateIndex = record->count; candidateIndex-- > 0; )
			{
				const b3MetalPairCandidate* candidate = metalCandidates + record->offset + candidateIndex;
				b3Shape* shapeA = b3Array_Get( world->shapes, candidate->shapeIndex );
				b3CreateContact( world, shapeA, shapeB, 0 );
			}
			continue;
		}
		int resultIndex = directMetalPlan ? (int)metalRecords[i].cpuFilterOffset : i;
#else
		int resultIndex = i;
#endif
		b3MoveResult* result = bp->moveResults + resultIndex;
		b3MovePair* pair = result->pairList;
		while ( pair != NULL )
		{
			int shapeIdA = pair->shapeIndexA;
			int shapeIdB = pair->shapeIndexB;
			int childIndex = pair->childIndex;

			b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
			b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );

			b3CreateContact( world, shapeA, shapeB, childIndex );

			if ( pair->heap )
			{
				b3MovePair* temp = pair;
				pair = pair->next;
				b3Free( temp, sizeof( b3MovePair ) );
			}
			else
			{
				pair = pair->next;
			}
		}
	}

	// Reset move buffer: clear only the bits that were set this step.
	// Invariant: bit set in movedProxies[type] iff proxyKey is present in moveArray.
	for ( int i = 0; i < bp->moveArray.count; ++i )
	{
		int proxyKey = bp->moveArray.data[i];
		b3ClearBit( &bp->movedProxies[B3_PROXY_TYPE( proxyKey )], B3_PROXY_ID( proxyKey ) );
	}
	b3Array_Clear( bp->moveArray );

	if ( bp->movePairs != NULL )
	{
		b3StackFree( alloc, bp->movePairs );
		bp->movePairs = NULL;
	}
	if ( bp->moveResults != NULL )
	{
		b3StackFree( alloc, bp->moveResults );
		bp->moveResults = NULL;
	}

	b3ValidateSolverSets( world );

	b3TracyCZoneEnd( create_contacts );

	b3TracyCZoneEnd( update_pairs );
}

bool b3BroadPhase_TestOverlap( const b3BroadPhase* bp, int proxyKeyA, int proxyKeyB )
{
	int typeIndexA = B3_PROXY_TYPE( proxyKeyA );
	int proxyIdA = B3_PROXY_ID( proxyKeyA );
	int typeIndexB = B3_PROXY_TYPE( proxyKeyB );
	int proxyIdB = B3_PROXY_ID( proxyKeyB );

	b3AABB aabbA = b3DynamicTree_GetAABB( bp->trees + typeIndexA, proxyIdA );
	b3AABB aabbB = b3DynamicTree_GetAABB( bp->trees + typeIndexB, proxyIdB );
	return b3AABB_Overlaps( aabbA, aabbB );
}

int b3BroadPhase_GetShapeIndex( b3BroadPhase* bp, int proxyKey )
{
	int typeIndex = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	return (int)b3DynamicTree_GetUserData( bp->trees + typeIndex, proxyId );
}

void b3ValidateBroadPhase( const b3BroadPhase* bp )
{
	b3DynamicTree_Validate( bp->trees + b3_dynamicBody );
	b3DynamicTree_Validate( bp->trees + b3_kinematicBody );

	// todo validate every shape AABB is contained in tree AABB
}

void b3ValidateNoEnlarged( const b3BroadPhase* bp )
{
#if B3_ENABLE_VALIDATION == 1
	for ( int j = 0; j < b3_bodyTypeCount; ++j )
	{
		const b3DynamicTree* tree = bp->trees + j;
		b3DynamicTree_ValidateNoEnlarged( tree );
	}
#else
	B3_UNUSED( bp );
#endif
}
