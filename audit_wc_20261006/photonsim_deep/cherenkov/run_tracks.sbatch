#!/bin/bash
#SBATCH --job-name=wc-cherenkov-tracks
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:35:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov/tracks-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov
source ../../photonsim_physics/geant4_env.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
g4_cherenkov_python=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$g4_cherenkov_python" python prepare_tracks.py
g4_cherenkov_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
source_list=()
for source_file in ../../photonsim_physics/upstream/src/*.cc; do
  if [[ "$source_file" != ../../photonsim_physics/upstream/src/DetectorConstruction.cc && "$source_file" != ../../photonsim_physics/upstream/src/SteppingAction.cc ]]; then source_list+=("$source_file"); fi
done
for variant in coarse dense; do
  detector_source=../../photonsim_physics/upstream/src/DetectorConstruction.cc
  if [[ "$variant" == dense ]]; then detector_source=DetectorConstruction_dense.cc; fi
  g++ -O2 -std=c++17 -I../../photonsim_physics/upstream/include -I"$g4_cherenkov_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) ../../photonsim_physics/headless_main.cc "${source_list[@]}" "$detector_source" SteppingAction_observe.cc -o "PhotonSim_$variant" -L"$g4_cherenkov_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
  for scenario in e025 e030 e035 e050 e1 e10 e100 mu60 mu100 mu1000 p500 p1000; do
    export AUDIT_STEPS="$PWD/${variant}_${scenario}_steps.csv" AUDIT_PHOTONS="$PWD/${variant}_${scenario}_photons.csv"
    /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_$variant" "$PWD/${variant}_${scenario}.mac" > "${variant}_${scenario}.log" 2>&1
    echo "Finished $variant $scenario"
  done
done
for scenario in e030 e050 e1 e10 mu60 mu100 p500; do
  export AUDIT_STEPS="$PWD/fine_${scenario}_steps.csv" AUDIT_PHOTONS="$PWD/fine_${scenario}_photons.csv"
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_dense" "$PWD/fine_${scenario}.mac" > "fine_${scenario}.log" 2>&1
  echo "Finished fine $scenario"
done
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python "$g4_cherenkov_python" python analyze_tracks.py
