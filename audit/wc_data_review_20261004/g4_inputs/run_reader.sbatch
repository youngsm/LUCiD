#!/bin/bash
#SBATCH --job-name=wc-g4-inputs
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/g4_inputs/reader-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/LUCiD
source audit/wc_data_review_20261004/geant4_physics/geant4_env.sh
set -u
g++ -std=c++17 audit/wc_data_review_20261004/g4_inputs/make_reader_fixture.cc \
  $(root-config --cflags --libs) -o audit/wc_data_review_20261004/g4_inputs/make_reader_fixture
bash audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/g4_inputs/make_reader_fixture \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/g4_inputs/reader_fixture.root
bash audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/g4_inputs/make_reader_fixture \
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/g4_inputs/nonzero_event_id.root 7
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
for g4_reader_script in check_reader.py reproduce_event_id.py; do
  /sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
    --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
    python "audit/wc_data_review_20261004/g4_inputs/$g4_reader_script"
done
