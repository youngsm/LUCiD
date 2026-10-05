#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --output=audit/wc_data_20261004/orchestration/env-%j.log
set -eu
cd /sdf/group/neutrino/youngsam/sim/LUCiD
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2
export PYTHONPATH=$PWD
export UV_CACHE_DIR=$PWD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --no-project --no-sync --python audit/wc_data_20261004/env_system/bin/python python -c 'import jax, numpy, flax, optax, h5py, uproot; from lucid.simulation import setup_event_simulator; print(jax.__version__, numpy.__version__, flax.__version__, jax.devices()); print("LUCiD import OK")'
