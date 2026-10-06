#!/bin/bash
#SBATCH --job-name=wc-transport-check
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport/final-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
source ../../photonsim_physics/geant4_env.sh
export UV_CACHE_DIR="$PWD/uv_cache" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
g++ -std=c++17 -fsyntax-only -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" minimal_water_capture_registration.cc
py=(/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python)
"${py[@]}" capture_regression.py analyze
"${py[@]}" analyze_em_channels.py
"${py[@]}" test_new_findings.py
