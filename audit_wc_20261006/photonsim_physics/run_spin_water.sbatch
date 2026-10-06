#!/bin/bash
#SBATCH --job-name=wc-spin-water
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_physics/spin-water-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_physics
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
source geant4_env.sh
export UV_CACHE_DIR="$PWD/uv_cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
g4_audit_uv=/sdf/home/y/youngsam/.local/bin/uv
g4_audit_python=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$g4_audit_uv" run --offline --no-project --python "$g4_audit_python" python spin_water_prepare.py
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
source_list=()
for source_file in upstream/src/*.cc; do
  if [[ "$source_file" != upstream/src/PhysicsList.cc && "$source_file" != upstream/src/SteppingAction.cc ]]; then source_list+=("$source_file"); fi
done
g++ -O2 -std=c++17 -Iupstream/include -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) headless_main.cc "${source_list[@]}" PhysicsList_spin.cc SteppingAction_spin_water.cc -o PhotonSim_spin_water -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_spin_water" "$PWD/pion_rest_water.mac" > pion_rest_water.log 2>&1
"$g4_audit_uv" run --offline --no-project --python "$g4_audit_python" python spin_regression.py analyze
