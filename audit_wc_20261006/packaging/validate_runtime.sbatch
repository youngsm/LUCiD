#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:ml-dev@milano
#SBATCH --qos=preemptable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=wc-runtime-package
#SBATCH --output=audit_wc_20261006/packaging/runtime-%j.log
set -euo pipefail
# Submit from the repository root. Probe binaries are built by primaries/run_spin.sbatch.
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
audit_primaries="$PWD/audit_wc_20261006/photonsim_deep/primaries"
audit_package="$PWD/audit_wc_20261006/packaging"
audit_python=${AUDIT_PYTHON:-/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python}
audit_uv=${AUDIT_UV:-/sdf/home/y/youngsam/.local/bin/uv}
audit_py=("$audit_uv" run --offline --no-project --python "$audit_python" python)
"${audit_py[@]}" "$audit_package/capture_environment.py"
export AUDIT_G4_ROOTFS="$audit_package/rootfs-validation-$SLURM_JOB_ID"
[[ ! -e "$AUDIT_G4_ROOTFS" ]]
"${audit_py[@]}" "$audit_primaries/prepare.py"
"${audit_py[@]}" "$audit_primaries/prepare_spin.py"
for variant in baseline repaired; do
  bash "$audit_package/run_photonsim.sh" "$audit_primaries/spin_probe_$variant" \
    "$audit_primaries/spin_random.commands" 256 "$audit_primaries/spin_$variant.tsv" \
    > "$audit_package/spin_${variant}_runtime.log" 2>&1
done
bash "$audit_package/run_photonsim.sh" "$audit_primaries/spin_probe_repaired" \
  "$audit_primaries/spin_fixed.commands" 256 "$audit_primaries/spin_beam_repaired.tsv" \
  > "$audit_package/spin_beam_runtime.log" 2>&1
bash "$audit_package/run_photonsim.sh" "$audit_primaries/spin_probe_repaired" \
  "$audit_primaries/correlated.commands" 5 "$audit_primaries/spin_missing_repaired.tsv" \
  > "$audit_package/spin_missing_runtime.log" 2>&1
"${audit_py[@]}" "$audit_primaries/analyze_spin.py"
printf 'PACKAGING_RUNTIME_PASS job=%s partition=%s rootfs=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$AUDIT_G4_ROOTFS"
