#!/bin/bash
#SBATCH --job-name=wc-deep-transport
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport/transport-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/transport
echo "Partition=$SLURM_JOB_PARTITION Job=$SLURM_JOB_ID Host=$(hostname)"
source ../../photonsim_physics/geant4_env.sh
export UV_CACHE_DIR="$PWD/uv_cache" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
g4_audit_source=../../photonsim_physics/upstream
g++ -O2 -std=c++17 -I"$g4_audit_source/include" -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) ledger_main.cc "$g4_audit_source"/src/*.cc -o PhotonSim_ledger -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
py=(/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python)
"${py[@]}" prepare_and_analyze.py prepare
wrapper=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh
audit_pids=()
while read -r case_name variant; do
  "$wrapper" "$PWD/PhotonSim_ledger" "$PWD/$case_name.mac" "$PWD/$case_name.csv" "$variant" > "$case_name.log" 2>&1 &
  audit_pids+=("$!")
  if [[ ${#audit_pids[@]} -eq 4 ]]; then
    for audit_pid in "${audit_pids[@]}"; do wait "$audit_pid"; done
    audit_pids=()
  fi
done < cases.txt
for audit_pid in "${audit_pids[@]}"; do wait "$audit_pid"; done
"${py[@]}" prepare_and_analyze.py analyze
