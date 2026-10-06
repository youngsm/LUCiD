#!/bin/bash
#SBATCH --job-name=wc-cherenkov-mean
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov/mean-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov
source ../../photonsim_physics/geant4_env.sh
g4_cherenkov_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
g++ -O2 -std=c++17 -I../../photonsim_physics/upstream/include -I"$g4_cherenkov_install/include/Geant4" -I"$wc_lcg_view/include" mean_yield.cc ../../photonsim_physics/upstream/src/DetectorConstruction.cc -o mean_yield -L"$g4_cherenkov_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP
./mean_yield
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --no-sync --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python analyze_mean.py
