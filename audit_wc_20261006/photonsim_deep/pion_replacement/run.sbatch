#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --job-name=wc_pion_replacement
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/pion_replacement/slurm-%j.out
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/pion_replacement
source ../../photonsim_physics/geant4_env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
audit_py=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
audit_uv=/sdf/home/y/youngsam/.local/bin/uv
export UV_CACHE_DIR="$PWD/uv_cache"
"$audit_uv" run --offline --no-project --python "$audit_py" python prepare.py
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
source_list=()
for source_file in ../../photonsim_physics/upstream/src/*.cc; do
  case "$source_file" in */SteppingAction.cc|*/EventAction.cc) ;; *) source_list+=("$source_file");; esac
done
for mode in on off; do
 g++ -O2 -std=c++17 -I. -I../../photonsim_physics/upstream/include -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) ../../photonsim_physics/headless_main.cc "${source_list[@]}" "SteppingAction_$mode.cc" EventAction_audit.cc -o "PhotonSim_$mode" -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
done
runner=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh
for energy in 2 200 1000 2000; do
 pids=()
 for mode in on off; do
  for pdg in 211 -211; do
   stem="${mode}_${pdg}_${energy}"
   AUDIT_PION_CSV="$PWD/$stem.csv" "$runner" "$PWD/PhotonSim_$mode" "$PWD/$stem.mac" > "$stem.log" 2>&1 &
   pids+=("$!")
  done
 done
 for pid in "${pids[@]}"; do wait "$pid"; done
 echo "Completed energy $energy"
done
"$audit_uv" run --offline --no-project --python "$audit_py" python analyze.py
