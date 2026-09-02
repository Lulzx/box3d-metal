#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
verify_root=$(mktemp -d "${TMPDIR:-/tmp}/box3d-metal-verify.XXXXXX")
checkout="$verify_root/box3d"

cleanup() {
    rm -rf "$verify_root"
}
trap cleanup EXIT HUP INT TERM

"$script_dir/bootstrap.sh" "$checkout"
"$checkout/build/metal-release/bin/test" MetalTest
"$checkout/build/metal-release/bin/metal_demo"
git -C "$checkout" diff --check

echo "clean reconstruction verified"
