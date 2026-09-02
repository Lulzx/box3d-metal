#!/bin/sh
set -eu

usage() {
    echo "usage: $0 [destination] [--no-build]" >&2
    exit 2
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

destination="${repo_dir}/box3d-metal-worktree"
build=1
if [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; then
    destination=$1
    shift
fi
if [ "$#" -gt 0 ]; then
    [ "$1" = "--no-build" ] || usage
    build=0
    shift
fi
[ "$#" -eq 0 ] || usage

. "$repo_dir/manifest.env"

for tool in git cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing required tool: $tool" >&2
        exit 1
    }
done

if [ -e "$destination" ]; then
    if [ ! -d "$destination" ] || [ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "destination must not exist or must be empty: $destination" >&2
        exit 1
    fi
else
    mkdir -p "$destination"
fi

git -C "$destination" clone --no-checkout "$BOX3D_REPOSITORY" .
git -C "$destination" checkout --detach "$BOX3D_BASELINE"

patch_file="$repo_dir/$BOX3D_METAL_PATCH"
git -C "$destination" apply --check "$patch_file"
git -C "$destination" apply "$patch_file"

if [ "$build" -eq 1 ]; then
    cmake -S "$destination" -B "$destination/build/metal-release" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBOX3D_METAL=ON \
        -DBOX3D_SAMPLES=OFF \
        -DBOX3D_UNIT_TESTS=ON \
        -DBOX3D_BENCHMARKS=ON
    cmake --build "$destination/build/metal-release"
fi

echo "Box3D Metal checkout ready at: $destination"
