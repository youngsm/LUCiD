#!/bin/bash
#SBATCH --job-name=wc-cherenkov-fine
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov/fine-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov
source ../../photonsim_physics/geant4_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
g4_cherenkov_python=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$g4_cherenkov_python" python prepare_tracks.py
for scenario in e030 e050 e1 e10 mu60 mu100 p500; do
  export AUDIT_STEPS="$PWD/fine_${scenario}_steps.csv" AUDIT_PHOTONS="$PWD/fine_${scenario}_photons.csv"
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_dense" "$PWD/fine_${scenario}.mac" > "fine_${scenario}.log" 2>&1
  echo "Finished fine $scenario"
done
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$g4_cherenkov_python" python analyze_mean.py
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$g4_cherenkov_python" python analyze_tracks.py
