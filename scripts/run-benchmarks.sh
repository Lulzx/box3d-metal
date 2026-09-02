#!/bin/sh
set -eu

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [box3d-metal-checkout]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
checkout=${1:-"$repo_dir/box3d-metal-worktree"}
build_dir="$checkout/build/metal-release"

cmake --build "$build_dir" --target \
    metal_fused_benchmark \
    metal_world_benchmark \
    metal_contact_benchmark \
    metal_mesh_benchmark \
    metal_joint_benchmark \
    metal_parallel_joint_benchmark

for executable in \
    metal_fused_benchmark \
    metal_world_benchmark \
    metal_contact_benchmark \
    metal_mesh_benchmark \
    metal_joint_benchmark \
    metal_parallel_joint_benchmark
do
    echo "===== $executable ====="
    "$build_dir/bin/$executable"
done
