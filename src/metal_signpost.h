// SPDX-FileCopyrightText: 2026 Box3D Metal contributors
// SPDX-License-Identifier: MIT

#pragma once

// Portable CPU-side signpost intervals for Metal step profiling.
//
// When BOX3D_SIGNPOSTS is enabled on Apple platforms this forwards to
// os_signpost so steps appear in Instruments. Everywhere else (or when the
// option is off) the macros compile to nothing and cost nothing.

#if defined( BOX3D_SIGNPOSTS ) && defined( __APPLE__ )
#include <os/signpost.h>

static inline os_log_t b3SignpostLog( void )
{
	static os_log_t log = NULL;
	if ( log == NULL )
	{
		log = os_log_create( "org.box3d", "metal-step" );
	}
	return log;
}

// os_signpost names must be string literals, so these stay macros. The SDK
// macros are variadic by design; a bare name leaves __VA_ARGS__ empty, which
// clang flags under -Werror, hence the scoped suppression inside the macro.
#define B3_SIGNPOST_BEGIN( name ) \
	_Pragma( "clang diagnostic push" ) \
	_Pragma( "clang diagnostic ignored \"-Wvariadic-macro-arguments-omitted\"" ) \
	os_signpost_interval_begin( b3SignpostLog(), OS_SIGNPOST_ID_EXCLUSIVE, name ) \
	_Pragma( "clang diagnostic pop" )
#define B3_SIGNPOST_END( name ) \
	_Pragma( "clang diagnostic push" ) \
	_Pragma( "clang diagnostic ignored \"-Wvariadic-macro-arguments-omitted\"" ) \
	os_signpost_interval_end( b3SignpostLog(), OS_SIGNPOST_ID_EXCLUSIVE, name ) \
	_Pragma( "clang diagnostic pop" )
#else
#define B3_SIGNPOST_BEGIN( name ) ( (void)0 )
#define B3_SIGNPOST_END( name ) ( (void)0 )
#endif
