#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc-genie-spin
#SBATCH --output=audit_wc_20261006/photonsim_deep/primaries/spin-%j.log
set -euo pipefail
# Submit from the checkout root, or supply AUDIT_REPO_ROOT explicitly.
cd "${AUDIT_REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
source audit_wc_20261006/photonsim_physics/geant4_env.sh
deep_dir="$PWD/audit_wc_20261006/photonsim_deep/primaries"
deep_upstream="$PWD/audit_wc_20261006/photonsim_physics/upstream"
deep_g4=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
deep_runtime=${AUDIT_G4_RUNNER:-"$PWD/audit_wc_20261006/packaging/run_photonsim.sh"}
deep_uv=${AUDIT_UV:-/sdf/home/y/youngsam/.local/bin/uv}
deep_python=${AUDIT_PYTHON:-/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python}
deep_py=("$deep_uv" run --offline --no-project --python "$deep_python" python)
"${deep_py[@]}" "$deep_dir/prepare.py"
"${deep_py[@]}" "$deep_dir/prepare_spin.py"
deep_common=("$deep_upstream/src/PrimaryGeneratorMessenger.cc" "$deep_upstream/src/DetectorConstruction.cc" "$deep_upstream/src/PhysicsList.cc")
for variant in baseline repaired; do
  if [[ $variant == baseline ]]; then deep_include="$deep_upstream/include"; deep_source="$deep_upstream/src"; else deep_include="$deep_dir/spin_patch/include"; deep_source="$deep_dir/spin_patch/src"; fi
  g++ -O2 -std=c++17 -I"$deep_include" -I"$deep_upstream/include" -I"$deep_g4/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) "$deep_dir/spin_probe.cc" "$deep_source/PrimaryGeneratorAction.cc" "$deep_source/RooTrackerReader.cc" "${deep_common[@]}" -o "$deep_dir/spin_probe_$variant" -L"$deep_g4/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
  "$deep_runtime" "$deep_dir/spin_probe_$variant" "$deep_dir/spin_random.commands" 256 "$deep_dir/spin_$variant.tsv" > "$deep_dir/spin_$variant.log" 2>&1
done
"$deep_runtime" "$deep_dir/spin_probe_repaired" "$deep_dir/spin_fixed.commands" 256 "$deep_dir/spin_beam_repaired.tsv" > "$deep_dir/spin_beam_repaired.log" 2>&1
"$deep_runtime" "$deep_dir/spin_probe_repaired" "$deep_dir/correlated.commands" 5 "$deep_dir/spin_missing_repaired.tsv" > "$deep_dir/spin_missing_repaired.log" 2>&1
"${deep_py[@]}" "$deep_dir/analyze_spin.py"
