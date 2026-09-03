include(GNUInstallDirs)

set(BOX3D_SOURCE_FILES
	aabb.c
	aabb.h
	algorithm.h
	arena_allocator.c
	arena_allocator.h
	bitset.c
	bitset.h
	block_allocator.c
	block_allocator.h
	body.c
	body.h
	broad_phase.c
	broad_phase.h
	capsule.c
	compound.c
	compound.h
	constraint_graph.c
	constraint_graph.h
	contact.c
	contact.h
	contact_solver.c
	contact_solver.h
	container.h
	convex_manifold.c
	core.c
	core.h
	ctz.h
	distance.c
	distance_joint.c
	dynamic_tree.c
	height_field.c
	hull.c
	hull.h
	id_pool.c
	id_pool.h
	island.c
	island.h
	joint.c
	joint.h
	manifold.c
	manifold.h
	math_functions.c
	math_internal.h
	metal_signpost.h
	metal_timeline.c
	metal_timeline.h
	mesh.c
	mesh_contact.c
	motor_joint.c
	mover.c
	name_cache.c
	name_cache.h
	parallel_for.c
	parallel_for.h
	parallel_joint.c
	physics_world.c
	physics_world.h
	platform.h
	prismatic_joint.c
	qsort.h
	rapidhash.h
	recording.c
	recording.h
	recording_ops.inl
	recording_replay.c
	recording_replay.h
	world_snapshot.c
	world_snapshot.h
	revolute_joint.c
	scheduler.c
	scheduler.h
	sensor.c
	sensor.h
	shape.c
	shape.h
	simd.c
	simd.h
	solver.c
	solver.h
	solver_set.c
	solver_set.h
	sphere.c
	spherical_joint.c
	table.c
	table.h
	timer.c
	triangle_manifold.c
	types.c
	verstable.h
	weld_joint.c
	wheel_joint.c

	box3d.natvis
)

set(BOX3D_INCLUDE_FILES
	../include/box3d/base.h
	../include/box3d/box3d.h
	../include/box3d/collision.h
	../include/box3d/config.h
	../include/box3d/constants.h
	../include/box3d/id.h
	../include/box3d/math_functions.h
	../include/box3d/metal.h
	../include/box3d/types.h
)

if(APPLE AND BOX3D_METAL)
	list(APPEND BOX3D_SOURCE_FILES
		metal_abi.h
		metal_backend.h
		metal_backend.m
		metal/box3d_contact.metal
		metal/box3d_main.metal
	)
	set(B3_METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal")
	set(B3_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/box3d_metallib.metallib")
	set(B3_BLOB_HEADER "${CMAKE_CURRENT_BINARY_DIR}/box3d_metallib_blob.h")
	set(B3_SOURCES_HEADER "${CMAKE_CURRENT_BINARY_DIR}/metal_sources.h")
	if(BOX3D_DOUBLE_PRECISION)
		set(B3_DOUBLE_FLAG 1)
	else()
		set(B3_DOUBLE_FLAG 0)
	endif()
	# FetchContent consumers never see the top-level option; keep the
	# runtime-compile fallback available for them.
	if(NOT DEFINED BOX3D_METAL_RUNTIME_COMPILE)
		set(BOX3D_METAL_RUNTIME_COMPILE ON)
	endif()
	# Precompile the merged MSL (main + contact, plus VF64 helpers for double
	# builds) into one metallib, then embed it as a byte blob loaded with
	# newLibraryWithData. The .metal files stay the single source of truth;
	# metal_sources.h regenerates the old runtime-compile strings from them.
	find_program(B3_XCRUN xcrun)
	if(B3_XCRUN)
		if(BOX3D_DOUBLE_PRECISION)
			set(B3_METAL_INPUTS "${B3_METAL_DIR}/box3d_main.metal" "${B3_METAL_DIR}/box3d_contact.metal")
			set(B3_METAL_DEPENDS "${B3_METAL_INPUTS}" "${CMAKE_CURRENT_SOURCE_DIR}/metal_vf64_ieee.metal")
			set(B3_METAL_FLAGS -DB3_DOUBLE_PRECISION)
		else()
			set(B3_METAL_INPUTS "${B3_METAL_DIR}/box3d_main.metal" "${B3_METAL_DIR}/box3d_contact.metal")
			set(B3_METAL_DEPENDS "${B3_METAL_INPUTS}")
			set(B3_METAL_FLAGS "")
		endif()
		add_custom_command(
			OUTPUT "${B3_METALLIB}"
			COMMAND "${B3_XCRUN}" -sdk macosx metal -O2 -I "${CMAKE_CURRENT_SOURCE_DIR}"
				${B3_METAL_FLAGS} ${B3_METAL_INPUTS} -o "${B3_METALLIB}"
			DEPENDS ${B3_METAL_DEPENDS} "${CMAKE_CURRENT_SOURCE_DIR}/metal_abi.h"
			COMMENT "Compiling Box3D Metal shaders"
			VERBATIM
		)
		set(B3_METALLIB_DEP "${B3_METALLIB}")
	else()
		if(NOT BOX3D_METAL_RUNTIME_COMPILE)
			message(FATAL_ERROR "xcrun not found and BOX3D_METAL_RUNTIME_COMPILE=OFF: no Metal shader source available")
		else()
			message(WARNING "xcrun not found; Metal backend will use runtime source compilation only")
		endif()
		set(B3_METALLIB_DEP "")
		set(B3_METALLIB "")
	endif()
	add_custom_command(
		OUTPUT "${B3_BLOB_HEADER}" "${B3_SOURCES_HEADER}"
		COMMAND "${CMAKE_COMMAND}"
			-DB3_METALLIB=${B3_METALLIB_DEP}
			-DB3_MAIN=${B3_METAL_DIR}/box3d_main.metal
			-DB3_CONTACT=${B3_METAL_DIR}/box3d_contact.metal
			-DB3_VF64=${CMAKE_CURRENT_SOURCE_DIR}/metal_vf64_ieee.metal
			-DB3_ABI=${CMAKE_CURRENT_SOURCE_DIR}/metal_abi.h
			-DB3_DOUBLE=${B3_DOUBLE_FLAG}
			-DB3_RUNTIME=${BOX3D_METAL_RUNTIME_COMPILE}
			-DB3_BLOB_OUT=${B3_BLOB_HEADER}
			-DB3_SRC_OUT=${B3_SOURCES_HEADER}
			-P "${CMAKE_CURRENT_SOURCE_DIR}/../tools/embed_metal.cmake"
		DEPENDS ${B3_METALLIB_DEP} "${B3_METAL_DIR}/box3d_main.metal" "${B3_METAL_DIR}/box3d_contact.metal"
			"${CMAKE_CURRENT_SOURCE_DIR}/metal_abi.h" "${CMAKE_CURRENT_SOURCE_DIR}/../tools/embed_metal.cmake"
		COMMENT "Embedding Box3D Metal metallib and fallback sources"
	)
	set(B3_GEN_HEADERS "${B3_BLOB_HEADER}" "${B3_SOURCES_HEADER}")
	add_custom_target(box3d_metal_shaders DEPENDS ${B3_GEN_HEADERS})
	set(B3_METAL_TARGET_DEFS "")
	if(B3_XCRUN)
		list(APPEND B3_METAL_TARGET_DEFS BOX3D_METAL_BLOB)
	endif()
	if(BOX3D_METAL_RUNTIME_COMPILE)
		list(APPEND B3_METAL_TARGET_DEFS BOX3D_METAL_RUNTIME_COMPILE)
	endif()
endif()

# Hide internal functions
# https://gcc.gnu.org/wiki/Visibility
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_library(box3d ${BOX3D_SOURCE_FILES} ${BOX3D_INCLUDE_FILES})

if(APPLE AND BOX3D_METAL)
	# Generated metallib blob + fallback sources must exist before compiling
	# metal_backend.m, which includes them.
	add_dependencies(box3d box3d_metal_shaders)
	if(B3_METAL_TARGET_DEFS)
		target_compile_definitions(box3d PRIVATE ${B3_METAL_TARGET_DEFS})
	endif()
endif()

# target_link_libraries(box3d PRIVATE eastl)

target_include_directories(box3d
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Box3D uses C17 for _Static_assert and anonymous unions. DEBUG_POSTFIX is set
# as a target property only so FetchContent consumers keep their own postfix.
set_target_properties(box3d PROPERTIES
	C_STANDARD 17
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS YES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
	DEBUG_POSTFIX "d"
)

if (BOX3D_COMPILE_WARNING_AS_ERROR)
	set_target_properties(box3d PROPERTIES COMPILE_WARNING_AS_ERROR ON)
endif()

if (BOX3D_PROFILE)
	target_compile_definitions(box3d PRIVATE BOX3D_PROFILE)
   	set(TRACY_DELAYED_INIT ON CACHE BOOL "Enable delayed init for Tracy" FORCE)
    set(TRACY_MANUAL_LIFETIME ON CACHE BOOL "Enable manual lifetime for Tracy" FORCE)

	FetchContent_Declare(
		tracy
		GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.13.1
		GIT_SHALLOW TRUE
		GIT_PROGRESS TRUE
	)
	FetchContent_MakeAvailable(tracy)

	target_link_libraries(box3d PUBLIC TracyClient)
endif()

if(BOX3D_VALIDATE)
	# Only effective when NDEBUG is undefined, see base.h
	message(STATUS "Box3D validation ON (debug builds)")
	target_compile_definitions(box3d PRIVATE BOX3D_VALIDATE)
endif()

if(BOX3D_SIGNPOSTS)
	message(STATUS "Box3D os_signposts ON")
	target_compile_definitions(box3d PRIVATE BOX3D_SIGNPOSTS)
endif()

if (BOX3D_DISABLE_SIMD)
	message(STATUS "Box3D SIMD disabled")
	target_compile_definitions(box3d PRIVATE BOX3D_DISABLE_SIMD)
endif()

if (BOX3D_DOUBLE_PRECISION)
	message(STATUS "Box3D double precision ON")
	# PUBLIC: consumers must see the same precision mode in the headers or they cannot match the ABI
	target_compile_definitions(box3d PUBLIC BOX3D_DOUBLE_PRECISION)
endif()

# Enable asserts in release with debug info
target_compile_definitions(box3d PUBLIC "$<$<CONFIG:RELWITHDEBINFO>:B3_ENABLE_ASSERT>")

if (MSVC)
	message(STATUS "Box3D on MSVC")
	if (BUILD_SHARED_LIBS)
		# this is needed by DLL users to import Box3D symbols
		target_compile_definitions(box3d INTERFACE BOX3D_DLL)
	endif()

	# Visual Studio won't load the natvis unless it is in the project
	target_sources(box3d PRIVATE box3d.natvis)

	# Warnings Wall is problematic in mixed C/C++, so using W4
	target_compile_options(box3d PRIVATE /W4)
	# target_compile_options(box3d PRIVATE /wd4820 /wd5045 /wd4061 /wd4711 /wd4514 /wd4365 /wd5219 /wd5039)
	target_compile_options(box3d PRIVATE /wd4820 /wd5045 /wd4061 /wd4711)
	# 4710 - warn about inline functions that are not inlined
	target_compile_options(box3d PRIVATE /wd4710 )

	if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
		target_compile_options(box3d PRIVATE -Wmissing-prototypes)
	endif()

elseif (MINGW)
	message(STATUS "Box3D on MinGW")
	target_compile_options(box3d PRIVATE -Wmissing-prototypes -Wall -Wextra -pedantic)
elseif (APPLE)
	message(STATUS "Box3D on Apple")
	target_compile_options(box3d PRIVATE -Wmissing-prototypes -Wall -Wextra -pedantic)
	if(BOX3D_METAL)
		message(STATUS "Box3D Metal compute backend ON")
		target_compile_definitions(box3d PUBLIC BOX3D_METAL)
		target_link_libraries(box3d PRIVATE "-framework Foundation" "-framework Metal")
	endif()
elseif (EMSCRIPTEN)
	message(STATUS "Box3D on Emscripten")
	target_compile_options(box3d PRIVATE -Wmissing-prototypes -Wall -Wextra -pedantic)
	# SSE2/wasm SIMD flags are set directory-wide in the top-level CMakeLists so the
	# tests, which include simd.h, get them too.
elseif (UNIX)
	message(STATUS "Box3D using Unix")
	target_compile_options(box3d PRIVATE -Wmissing-prototypes -Wall -Wextra -pedantic -Wno-unused-value)
	if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
		# raspberry pi
		# -mfpu=neon
		# target_compile_options(box3d PRIVATE)
	else()
	endif()
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src" FILES ${BOX3D_SOURCE_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../include" PREFIX "include" FILES ${BOX3D_INCLUDE_FILES})

add_library(box3d::box3d ALIAS box3d)

# libm is required on many Unix toolchains for sqrtf, etc.
if(UNIX AND NOT APPLE AND NOT EMSCRIPTEN)
  target_link_libraries(box3d PUBLIC m)
endif()

install(
  TARGETS box3d
  EXPORT box3dConfig
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
  EXPORT box3dConfig
  NAMESPACE box3d::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/box3d"
)

install(
  FILES ${BOX3D_INCLUDE_FILES}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/box3d
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/box3dConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/box3dConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/box3d"
)
