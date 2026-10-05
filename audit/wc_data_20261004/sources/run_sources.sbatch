#!/bin/bash
#SBATCH --job-name=wc-source-repro
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/sources/repro-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/LUCiD
source audit/wc_data_20261004/sources/geant4_env.sh
set -u
wc_audit_python=/sdf/group/neutrino/youngsam/.conda/envs/py310_torch/bin/python
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --no-project --python "$wc_audit_python" audit/wc_data_20261004/sources/source_repro.py prepare
cd audit/wc_data_20261004/sources
./build_headless/PhotonSim gamma_prompt.mac > gamma_prompt.log 2>&1
./build_headless/PhotonSim gamma_delayed.mac > gamma_delayed.log 2>&1
./build_headless/PhotonSim pion_deflection.mac > pion_deflection.log 2>&1
./build_headless/PhotonSim genie_isotropic.mac > genie_isotropic.log 2>&1
/sdf/home/y/youngsam/.local/bin/uv run --no-project --python "$wc_audit_python" source_repro.py analyze
