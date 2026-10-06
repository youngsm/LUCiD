#!/bin/bash
#SBATCH --job-name=wc-cherenkov-minimal
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov/minimal-%j.log
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/lucid-data/LUCID/audit_wc_20261006/photonsim_deep/cherenkov
for scenario in default small; do
  /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_review_20261004/geant4_physics/run_photonsim.sh "$PWD/../../photonsim_physics/PhotonSim_v102" "$PWD/min_${scenario}.mac" > "min_${scenario}.log" 2>&1
  tail -6 "min_${scenario}.log"
done
