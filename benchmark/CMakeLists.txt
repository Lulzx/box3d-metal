# Benchmark app

set(BOX3D_BENCHMARK_FILES
	main.c
)

add_executable(benchmark ${BOX3D_BENCHMARK_FILES})

set_target_properties(benchmark PROPERTIES C_STANDARD 17)

target_link_libraries(benchmark PRIVATE box3d shared)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${BOX3D_BENCHMARK_FILES})

if(APPLE AND BOX3D_METAL)
	# These two kernel microbenchmarks deliberately use the private backend API.
	# Keep that implementation boundary hidden from shared-library consumers.
	if(NOT BUILD_SHARED_LIBS)
		add_executable(metal_benchmark metal_integration.c)
		set_target_properties(metal_benchmark PROPERTIES C_STANDARD 17)
		target_include_directories(metal_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
		target_link_libraries(metal_benchmark PRIVATE box3d)

		add_executable(metal_fused_benchmark metal_fused.c)
		set_target_properties(metal_fused_benchmark PROPERTIES C_STANDARD 17)
		target_include_directories(metal_fused_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
		target_link_libraries(metal_fused_benchmark PRIVATE box3d)
	endif()

	add_executable(metal_world_benchmark metal_world.c)
	set_target_properties(metal_world_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_world_benchmark PRIVATE box3d)

	add_executable(metal_contact_benchmark metal_contacts.c)
	set_target_properties(metal_contact_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_contact_benchmark PRIVATE box3d)

	add_executable(metal_resident_contact_benchmark metal_resident_contacts.c)
	set_target_properties(metal_resident_contact_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_resident_contact_benchmark PRIVATE box3d)

	add_executable(metal_mesh_benchmark metal_mesh_contacts.c)
	set_target_properties(metal_mesh_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_mesh_benchmark PRIVATE box3d)

	add_executable(metal_joint_benchmark metal_distance_joints.c)
	set_target_properties(metal_joint_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_joint_benchmark PRIVATE box3d)

	add_executable(metal_parallel_joint_benchmark metal_parallel_joints.c)
	set_target_properties(metal_parallel_joint_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_parallel_joint_benchmark PRIVATE box3d)

	add_executable(metal_demo metal_demo.c)
	set_target_properties(metal_demo PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_demo PRIVATE box3d)

	add_executable(metal_scene_benchmark metal_scene_benchmark.c)
	set_target_properties(metal_scene_benchmark PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_scene_benchmark PRIVATE box3d shared)

	add_executable(metal_bandwidth metal_bandwidth.m)
	set_target_properties(metal_bandwidth PROPERTIES C_STANDARD 17)
	target_link_libraries(metal_bandwidth PRIVATE box3d)
	if(APPLE)
		target_link_libraries(metal_bandwidth PRIVATE "-framework Foundation" "-framework Metal")
	endif()
endif()
