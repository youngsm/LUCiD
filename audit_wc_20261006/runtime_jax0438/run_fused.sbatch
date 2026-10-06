#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=wc_jax0438_fused
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/runtime_jax0438/fused-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
[[ "$SLURM_JOB_PARTITION" == milano || "$SLURM_JOB_PARTITION" == roma ]]
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
audit_runtime="$PWD/audit_wc_20261006/runtime_jax0438"
"$audit_runtime/venv/bin/python" "$audit_runtime/check_fused_reflections.py"
