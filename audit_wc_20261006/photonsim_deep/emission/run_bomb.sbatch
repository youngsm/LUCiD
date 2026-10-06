#!/bin/bash
#SBATCH --job-name=wc-g4-emission-bomb
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission/bomb-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission
source ../../photonsim_physics/geant4_env.sh
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export UV_CACHE_DIR="$PWD/uv_cache"
emission_uv=/sdf/home/y/youngsam/.local/bin/uv
emission_py=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$emission_uv" run --offline --no-project --python "$emission_py" python prepare_bomb.py
for mode in stream buffer; do
    emission_tag="bomb_${mode}"
    export AUDIT_EMISSION_BINARY="$PWD/$emission_tag.bin"
    export AUDIT_EMISSION_SUMMARY="$PWD/$emission_tag.jsonl"
    /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_observed" "$PWD/$emission_tag.mac" > "$emission_tag.log" 2>&1
done
"$emission_uv" run --offline --no-project --python "$emission_py" python analyze.py
