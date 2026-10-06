#!/bin/bash
#SBATCH --job-name=wc-radiative-check
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport/radiative-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
export UV_CACHE_DIR="$PWD/uv_cache" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
py=(/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python)
"${py[@]}" test_radiative_oracle.py
"${py[@]}" prepare_and_analyze.py analyze
"${py[@]}" analyze_em_channels.py
