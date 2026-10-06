#!/bin/bash
#SBATCH --job-name=wc-bomb-audit
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/end_to_end/bomb-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export UV_CACHE_DIR="$PWD/audit_wc_20261006/end_to_end/uv_cache"
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/end_to_end/prepare_bomb.py
/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/audit_wc_20261006/photonsim_physics/PhotonSim_v102" "$PWD/audit_wc_20261006/end_to_end/bomb_source/bomb.mac" > audit_wc_20261006/end_to_end/bomb_source/geant4.log 2>&1
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python audit_wc_20261006/end_to_end/check_actual_production.py bomb
