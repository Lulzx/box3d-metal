# Box3D unit test app

set(BOX3D_TEST_FILES
    main.c
	test_allocator.c
    test_bitset.c
	test_body.c
	test_body_query.c
	test_collision.c
	test_compound.c
	test_container.c
    test_determinism.c
    test_distance.c
    test_hash.c
    test_height_field.c
    test_hull.c
    test_id.c
    test_joint.c
    test_large_world.c
    test_macros.h
    test_manifold.c
    test_math.c
    test_mesh.c
	test_mover.c
    test_name_cache.c
    test_recording.c
    test_sat.c
    test_shape.c
    test_table.c
    test_world.c
)

if(APPLE AND BOX3D_METAL)
	list(APPEND BOX3D_TEST_FILES test_metal.c)
endif()

add_executable(box3d_test ${BOX3D_TEST_FILES})

set_target_properties(box3d_test PROPERTIES
    C_STANDARD 17
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    OUTPUT_NAME test
)

# Special access to Box3D internals for testing
target_include_directories(box3d_test PRIVATE ${CMAKE_SOURCE_DIR}/src)

if(BOX3D_VALIDATE)
	target_compile_definitions(box3d_test PRIVATE BOX3D_VALIDATE)
endif()

if(BOX3D_DISABLE_SIMD)
	target_compile_definitions(box3d_test PRIVATE BOX3D_DISABLE_SIMD)
endif()

target_link_libraries(box3d_test PRIVATE box3d shared)

# Lets `ctest --output-on-failure` work from the build dir.
# OUTPUT_NAME keeps the binary at bin/test for existing scripts.
add_test(NAME box3d_unit COMMAND box3d_test)
set_tests_properties(box3d_unit PROPERTIES TIMEOUT 120)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "" FILES ${BOX3D_TEST_FILES})
