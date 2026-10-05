#!/bin/bash
#SBATCH --job-name=wc-g4-audit
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/repro-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics
source geant4_env.sh
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
g++ -O2 -std=c++17 -IPhotonSim/include -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) headless_main.cc PhotonSim/src/*.cc -o PhotonSim_audit -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
g4_audit_uv=/sdf/home/y/youngsam/.local/bin/uv
g4_audit_python=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python
"$g4_audit_uv" run --offline --no-project --python "$g4_audit_python" python source_repro.py prepare
./run_photonsim.sh "$PWD/PhotonSim_audit" "$PWD/gamma_prompt.mac" > gamma_prompt.log 2>&1
./run_photonsim.sh "$PWD/PhotonSim_audit" "$PWD/gamma_delayed.mac" > gamma_delayed.log 2>&1
./run_photonsim.sh "$PWD/PhotonSim_audit" "$PWD/pion_deflection.mac" > pion_deflection.log 2>&1
./run_photonsim.sh "$PWD/PhotonSim_audit" "$PWD/genie_isotropic.mac" > genie_isotropic.log 2>&1
"$g4_audit_uv" run --offline --no-project --python "$g4_audit_python" python source_repro.py analyze
