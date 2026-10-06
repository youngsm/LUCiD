#!/bin/bash
#SBATCH --job-name=wc-g4-step-tag
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission/step-tag-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/emission
source ../../photonsim_physics/geant4_env.sh
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export UV_CACHE_DIR="$PWD/uv_cache"
emission_uv=/sdf/home/y/youngsam/.local/bin/uv
emission_py=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$emission_uv" run --offline --no-project --python "$emission_py" python prepare_step_tag.py
emission_g4=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
emission_up=../../photonsim_physics/upstream
emission_sources=()
for source_file in "$emission_up"/src/*.cc; do
  case "$source_file" in */SteppingAction.cc|*/DataManager.cc) ;; *) emission_sources+=("$source_file");; esac
done
g++ -O2 -std=c++17 -Istep_tag -I"$emission_up/include" -I"$emission_g4/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) \
  ../../photonsim_physics/headless_main.cc "${emission_sources[@]}" step_tag/SteppingAction.cc step_tag/DataManager.cc \
  -o PhotonSim_step_tag -L"$emission_g4/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
for scenario in gamma electron pion muon; do
  for mode in baseline fixed; do
    emission_tag="${scenario}_tag_${mode}"
    export AUDIT_STEP_BINARY="$PWD/$emission_tag.bin"
    if [[ "$mode" == fixed ]]; then export AUDIT_EXACT_STEP=1; else unset AUDIT_EXACT_STEP; fi
    /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/PhotonSim_step_tag" "$PWD/$emission_tag.mac" > "$emission_tag.log" 2>&1
  done
done
"$emission_uv" run --offline --no-project --python "$emission_py" python analyze_step_tag.py
