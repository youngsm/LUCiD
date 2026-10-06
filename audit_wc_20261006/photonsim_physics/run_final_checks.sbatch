#!/bin/bash
#SBATCH --job-name=wc-source-checks
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_physics/final-checks-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_physics
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
export UV_CACHE_DIR="$PWD/uv_cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
py=(/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python)
"${py[@]}" production_chain_presence.py
set +e
"${py[@]}" test_spin_physics.py baseline
baseline_status=$?
"${py[@]}" test_segment_counts.py
segment_status=$?
set -e
test "$baseline_status" -eq 1
test "$segment_status" -eq 1
"${py[@]}" test_spin_physics.py water
