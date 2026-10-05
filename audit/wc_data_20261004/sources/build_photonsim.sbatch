#!/bin/bash
#SBATCH --job-name=wc-source-build
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/sources/build-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/LUCiD
source audit/wc_data_20261004/sources/geant4_env.sh
set -u
cmake -S audit/wc_data_20261004/sources/headless -B audit/wc_data_20261004/sources/build_headless -DCMAKE_BUILD_TYPE=Release -DGeant4_DIR=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt/lib64/Geant4-11.3.0
cmake --build audit/wc_data_20261004/sources/build_headless -j 4
