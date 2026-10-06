#!/bin/bash
# Build an empty Apptainer bind-mount sandbox; copy no host identity files.
set -euo pipefail
audit_packaging_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit_rootfs=${1:-${AUDIT_G4_ROOTFS:-"$audit_packaging_dir/rootfs"}}
mkdir -p -- "$audit_rootfs"
for audit_dir in bin usr lib lib64 etc cvmfs sdf tmp dev proc sys run var build; do
  mkdir -p -- "$audit_rootfs/$audit_dir"
done
mkdir -p -- "$audit_rootfs/build/jenkins/workspace/lcg_release_pipeline/build/projects/ROOT-6.34.02/src/ROOT-6.34.02-build/lib"
printf '%s\n' "$audit_rootfs"
