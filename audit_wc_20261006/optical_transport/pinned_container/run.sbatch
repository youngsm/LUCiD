#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc_pinned_jax
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/optical_transport/pinned_container/slurm-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export APPTAINERENV_PYTHONPATH=/opt/LUCiD
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_JAX_PLATFORMS=cpu
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_SLURM_JOB_PARTITION="$SLURM_JOB_PARTITION"
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
apptainer exec --cleanenv --bind /sdf:/sdf,/fs:/fs,"$PWD":/opt/LUCiD \
    --pwd /opt/LUCiD /sdf/data/neutrino/zhenxiong/software/images/lucid.sif \
    /opt/conda/bin/python audit_wc_20261006/optical_transport/pinned_container/check_environment.py
