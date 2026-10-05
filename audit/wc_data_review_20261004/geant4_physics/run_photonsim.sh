#!/bin/bash
set -euo pipefail
g4_audit_base=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics
g4_audit_rootlib=/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.34.02-18eb6/x86_64-el8-gcc11-opt/lib
apptainer exec --bind /bin,/usr,/lib,/lib64,/etc,/cvmfs,/sdf,/tmp --bind "$g4_audit_rootlib:/build/jenkins/workspace/lcg_release_pipeline/build/projects/ROOT-6.34.02/src/ROOT-6.34.02-build/lib" "$g4_audit_base/rootfs" /bin/bash -c 'source "$1/geant4_env.sh"; shift; exec "$@"' audit "$g4_audit_base" "$@"
