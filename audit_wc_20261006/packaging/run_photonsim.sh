#!/bin/bash
# Configurable replacement for the original external audit runtime wrapper.
# This is a host/CVMFS bind-mount sandbox, not a portable Geant4 container.
set -euo pipefail
case ${SLURM_JOB_PARTITION:-} in
  milano|roma) ;;
  *) printf '%s\n' 'Run the Geant4 audit inside a milano or roma CPU Slurm allocation.' >&2; exit 2 ;;
esac
if [[ $# -lt 1 ]]; then
  printf '%s\n' 'Usage: run_photonsim.sh BINARY [ARG ...]' >&2
  exit 2
fi
audit_packaging_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit_repo=${AUDIT_REPO_ROOT:-$(cd -- "$audit_packaging_dir/../.." && pwd)}
audit_env=${AUDIT_G4_ENV:-"$audit_repo/audit_wc_20261006/photonsim_physics/geant4_env.sh"}
audit_rootfs=${AUDIT_G4_ROOTFS:-"$audit_packaging_dir/rootfs"}
audit_rootlib=${AUDIT_ROOT_LIB:-/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.34.02-18eb6/x86_64-el8-gcc11-opt/lib}
audit_apptainer=${AUDIT_APPTAINER:-apptainer}
[[ -r "$audit_env" ]] || { printf 'Missing Geant4 setup: %s\n' "$audit_env" >&2; exit 2; }
[[ -d "$audit_rootlib" ]] || { printf 'Missing ROOT libraries: %s\n' "$audit_rootlib" >&2; exit 2; }
bash "$audit_packaging_dir/make_rootfs.sh" "$audit_rootfs" >/dev/null
exec "$audit_apptainer" exec \
  --bind /bin,/usr,/lib,/lib64,/etc,/cvmfs,/sdf,/tmp \
  --bind "$audit_repo:$audit_repo" \
  --bind "$audit_env:$audit_env:ro" \
  --bind "$audit_rootlib:/build/jenkins/workspace/lcg_release_pipeline/build/projects/ROOT-6.34.02/src/ROOT-6.34.02-build/lib" \
  "$audit_rootfs" /bin/bash -c 'source "$1"; shift; exec "$@"' audit "$audit_env" "$@"
