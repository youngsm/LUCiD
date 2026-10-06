#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=wc-deep-primary
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/primaries/job-%j.log
set -euo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
source audit_wc_20261006/photonsim_physics/geant4_env.sh
deep_dir="$PWD/audit_wc_20261006/photonsim_deep/primaries"
deep_upstream="$PWD/audit_wc_20261006/photonsim_physics/upstream"
deep_g4=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
deep_runtime=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh
deep_py=(/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python)
"${deep_py[@]}" "$deep_dir/prepare.py"
g++ -O2 -std=c++17 -I"$deep_upstream/include" -I"$deep_g4/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) "$deep_dir/primary_probe.cc" "$deep_upstream/src/PrimaryGeneratorAction.cc" "$deep_upstream/src/PrimaryGeneratorMessenger.cc" "$deep_upstream/src/RooTrackerReader.cc" "$deep_upstream/src/DetectorConstruction.cc" "$deep_upstream/src/PhysicsList.cc" -o "$deep_dir/primary_probe" -L"$deep_g4/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/bomb.commands" 20000 "$deep_dir/bomb.tsv" > "$deep_dir/bomb.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/bomb.commands" 10 "$deep_dir/bomb_repeat.tsv" > "$deep_dir/bomb_repeat.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/bomb_newseed.commands" 10 "$deep_dir/bomb_newseed.tsv" > "$deep_dir/bomb_newseed.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/multigun.commands" 5 "$deep_dir/multigun.tsv" > "$deep_dir/multigun.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/correlated.commands" 20000 "$deep_dir/correlated.tsv" > "$deep_dir/correlated.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/ion.commands" 1 "$deep_dir/ion.tsv" > "$deep_dir/ion.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/ion.commands" 1 "$deep_dir/ion_preload.tsv" preload-ion > "$deep_dir/ion_preload.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/missing.commands" 1 "$deep_dir/missing.tsv" > "$deep_dir/missing.log" 2>&1
"$deep_runtime" "$deep_dir/primary_probe" "$deep_dir/wrong_tree.commands" 1 "$deep_dir/wrong_tree.tsv" > "$deep_dir/wrong_tree.log" 2>&1
"${deep_py[@]}" "$deep_dir/analyze.py"
