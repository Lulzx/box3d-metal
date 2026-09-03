#!/usr/bin/env bash
# Use this to build Box3D on any system with a bash shell.
# Non-destructive: reuses the existing build dir and fails fast.
set -euo pipefail

cmake -S . -B build
cmake --build build
