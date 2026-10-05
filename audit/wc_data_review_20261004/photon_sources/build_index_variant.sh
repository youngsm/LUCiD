#!/bin/bash
set -eo pipefail
cd /sdf/group/neutrino/youngsam/sim/LUCiD
source audit/wc_data_review_20261004/geant4_physics/geant4_env.sh
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python python - <<'PY'
import importlib.util
from pathlib import Path
import re
import numpy as np
base = Path('audit/wc_data_review_20261004/photon_sources')
spec = importlib.util.spec_from_file_location('check_cherenkov_index', base / 'check_cherenkov_index.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original = m.PHOTON_SOURCE.read_text()
water = original.split('G4Material* DetectorConstruction::ConstructWater()')[1].split('G4Material* DetectorConstruction::ConstructLiquidArgon()')[0]
energy = m.cpp_array(water, 'photonEnergy', 'nEntries')
index = m.measured_n(1239.8419843320026 / energy)
pattern = r'(G4double refractiveIndex\[nEntries\] = \{)[^}]+(\};)'
modified = re.sub(pattern, lambda mt: mt.group(1) + '\n    ' + ', '.join(f'{n:.12f}' for n in index) + '\n  ' + mt.group(2), original, count=1)
(base / 'DetectorConstruction_measured.cc').write_text(modified)
print(index.tolist())
PY
g4_audit_install=/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt
source_base=audit/wc_data_20261004/sources/PhotonSim-v1.0.0
source_list=()
for source_cc in "$source_base"/src/*.cc; do
  if [[ "$source_cc" != "$source_base/src/DetectorConstruction.cc" ]]; then source_list+=("$source_cc"); fi
done
g++ -O2 -std=c++17 -I"$source_base/include" -I"$g4_audit_install/include/Geant4" -I"$wc_lcg_view/include" $(root-config --cflags) audit/wc_data_review_20261004/geant4_physics/headless_main.cc "${source_list[@]}" audit/wc_data_review_20261004/photon_sources/DetectorConstruction_measured.cc -o audit/wc_data_review_20261004/photon_sources/PhotonSim_measured -L"$g4_audit_install/lib64" -lG4run -lG4analysis -lG4ptl -pthread -lG4physicslists -lG4processes -lG4event -lG4tracking -lG4track -lG4particles -lG4geometry -lG4materials -lG4intercoms -lG4global -lG4digits_hits -lG4graphics_reps -L"$wc_lcg_view/lib" -lCLHEP $(root-config --libs)
