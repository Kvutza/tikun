#!/usr/bin/env bash
set -euo pipefail

echo "running tikun verification suite via buck2..."
./tools/bin/buck2 test //:tikun_core_tests //:tikun_cpu_tests --target-platforms=//:macos-arm64-platform

echo "building tikun python extension via buck2..."
./tools/bin/buck2 build //:tikun_py --target-platforms=//:macos-arm64-platform

echo "all verification checks passed."

