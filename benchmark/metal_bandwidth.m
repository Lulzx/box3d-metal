// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

// Phase 0 bandwidth probe: measures achievable Metal device memory bandwidth
// with a large buffer-to-buffer copy so solver throughput can be evaluated
// against the roofline (see tools/metal_roofline.py). Prints one CSV row:
//   bytes, repeats, median_gbps
// Override size/repeats with BOX3D_METAL_BANDWIDTH_BYTES / _REPEATS.

#include "box3d/box3d.h"

#include <stdio.h>
#include <stdlib.h>

#if defined( __APPLE__ )
#include <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static double MonotonicMs( void )
{
	struct timespec ts;
	clock_gettime( CLOCK_MONOTONIC, &ts );
	return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int CompareDouble( const void* a, const void* b )
{
	double x = *(const double*)a, y = *(const double*)b;
	return ( x > y ) - ( x < y );
}

int main( void )
{
	@autoreleasepool
	{
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if ( device == nil )
		{
			fprintf( stderr, "no Metal device\n" );
			return 1;
		}
		const char* bytesText = getenv( "BOX3D_METAL_BANDWIDTH_BYTES" );
		const char* repeatsText = getenv( "BOX3D_METAL_BANDWIDTH_REPEATS" );
		NSUInteger bytes = bytesText != NULL ? (NSUInteger)strtoull( bytesText, NULL, 10 ) : 256u * 1024u * 1024u;
		int repeats = repeatsText != NULL ? atoi( repeatsText ) : 20;
		if ( bytes < 1024 * 1024 || repeats < 1 )
		{
			fprintf( stderr, "bytes must be >= 1MB and repeats >= 1\n" );
			return 2;
		}
		id<MTLBuffer> src = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		id<MTLBuffer> dst = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
		id<MTLBuffer> back = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
		if ( src == nil || dst == nil || back == nil )
		{
			fprintf( stderr, "buffer allocation failed\n" );
			return 1;
		}
		memset( src.contents, 0xAB, bytes );
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if ( queue == nil )
		{
			fprintf( stderr, "queue creation failed\n" );
			return 1;
		}
		double* samples = malloc( (size_t)repeats * sizeof( double ) );
		if ( samples == NULL )
		{
			return 1;
		}
		for ( int i = 0; i < repeats; ++i )
		{
			id<MTLCommandBuffer> cb = [queue commandBuffer];
			id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
			[blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
			[blit copyFromBuffer:dst sourceOffset:0 toBuffer:back destinationOffset:0 size:bytes];
			[blit endEncoding];
			double start = MonotonicMs();
			[cb commit];
			[cb waitUntilCompleted];
			double elapsed = MonotonicMs() - start;
			if ( cb.status != MTLCommandBufferStatusCompleted || elapsed <= 0.0 )
			{
				free( samples );
				fprintf( stderr, "copy failed\n" );
				return 1;
			}
			// Two full traversals (shared->private, private->shared).
			samples[i] = 2.0 * (double)bytes / ( elapsed / 1000.0 ) / 1e9;
		}
		qsort( samples, (size_t)repeats, sizeof( double ), CompareDouble );
		printf( "bytes,repeats,median_gbps\n" );
		printf( "%llu,%d,%.2f\n", (unsigned long long)bytes, repeats, samples[repeats / 2] );
		printf( "# device=%s\n", device.name.UTF8String );
		free( samples );
		[src release];
		[dst release];
		[back release];
		[queue release];
	}
	return 0;
}
#else
int main( void )
{
	fprintf( stderr, "metal_bandwidth requires Apple Metal\n" );
	return 1;
}
#endif
