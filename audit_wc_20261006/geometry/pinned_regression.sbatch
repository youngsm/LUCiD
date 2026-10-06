#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=wc_pinned_regression
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/geometry/pinned-regression-%j.out
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
interpreter="$PWD/audit_wc_20261006/runtime_jax0438/venv/bin/python"
"$interpreter" audit_wc_20261006/geometry/repro_reflection_selfhit_pinned.py
"$interpreter" audit_wc_20261006/geometry/repro_reflection_selfhit_pinned.py --fixed
