#!/bin/sh
set -e
cd "$(dirname "$0")"
AIR_LIST=""
i=0
for f in Kernels/*.metal Shaders/*.metal; do
  [ -f "$f" ] || continue
  out="/tmp/gh_kernel_${i}.air"
  xcrun -sdk macosx metal -c "$f" -o "$out"
  AIR_LIST="$AIR_LIST $out"
  i=$((i + 1))
done
# shellcheck disable=SC2086
xcrun -sdk macosx metallib $AIR_LIST -o Kernels/default.metallib
echo "Wrote Kernels/default.metallib"
if [ -n "${CONFIGURATION_BUILD_DIR}" ]; then
  cp -f Kernels/default.metallib "${CONFIGURATION_BUILD_DIR}/default.metallib"
  echo "Copied default.metallib to ${CONFIGURATION_BUILD_DIR}"
fi
