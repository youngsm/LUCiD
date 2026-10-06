#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc_optics_audit
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/optical_transport/slurm-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4'
/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python audit_wc_20261006/optical_transport/check_transport_physics.py
