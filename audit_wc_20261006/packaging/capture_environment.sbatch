#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=wc-runtime-env
#SBATCH --output=audit_wc_20261006/packaging/environment-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
audit_python=${AUDIT_PYTHON:-/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python}
audit_uv=${AUDIT_UV:-/sdf/home/y/youngsam/.local/bin/uv}
"$audit_uv" run --offline --no-project --python "$audit_python" python audit_wc_20261006/packaging/capture_environment.py
