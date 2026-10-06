# Audit runtime setup

This directory makes the Geant4 wrapper independent of the earlier audit
directory. It does **not** contain a portable operating-system image, Geant4,
ROOT, Python environment, or large raw simulation files. Source launchers and
recorded results have not all been tested on a relocated checkout.

Run every build, Python import, numerical analysis and test inside a **milano or
roma CPU Slurm allocation**. The wrapper enforces those partition names. The
audit used no GPUs. Submit the example jobs from the repository root.

## Exact audited external software

The included `../photonsim_physics/geant4_env.sh` sources these CVMFS installs:

- GCC 11.3.0: `/cvmfs/sft.cern.ch/lcg/releases/gcc/11.3.0/x86_64-el8/setup.sh`.
- LCG 107 view: `/cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el8-gcc11-opt`.
- Geant4 11.3.0, including its data: `/cvmfs/sft.cern.ch/lcg/releases/Geant4/11.3.0-3cd7f/x86_64-el8-gcc11-opt`.
- ROOT 6.34.02: `/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.34.02-18eb6/x86_64-el8-gcc11-opt`.
- CLHEP 2.4.7.1 is resolved through that LCG view.
- A working host Apptainer installation is required.

The original baseline Python interpreter is
`/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python`.
Its `pyvenv.cfg` says CPython 3.12.8, `include-system-site-packages=true`, with
base interpreter under `/sdf/group/neutrino/youngsam/.conda/envs/py310_torch/bin`.
The directory name does not identify its actual Python version. The original
runner is `/sdf/home/y/youngsam/.local/bin/uv`. Copying the virtual environment
alone does not reconstruct this inherited environment. The packaging validation
captures its visible package versions as `requirements-baseline-resolved.txt`
and selected core modules in `environment-baseline.json`. This inventory is not
a portable environment lock or proof that every inherited package is needed.
`uv pip freeze --python` alone lists only the overlay in this setup, so the
included `capture_environment.py` explicitly includes inherited metadata.
The complete metadata/core-import capture succeeded in milano job **39974575**;
see `environment-39974575.log` and `environment-baseline.json`.

The isolated JAX 0.4.38 environment is different. Its exact captured Python
package versions are in `../runtime_jax0438/requirements-resolved.txt`, with
runtime metadata in `../runtime_jax0438/environment.json`. Create a clean
Python 3.12 environment and install from that requirements file on an allowed
allocation. The recorded six-million-photon charge study used JAX 0.11 and must
not be relabelled as a JAX 0.4.38 result. No complete production-container
equivalence is claimed.

## Geant4 wrapper and sandbox

`run_photonsim.sh BINARY [ARG ...]` derives the repository location from its own
path and runs the requested executable inside an Apptainer bind-mount sandbox.
`make_rootfs.sh [PATH]` creates only empty directories; it copies no host passwd,
group, DNS, or other identity files. The host `/etc` is bound at execution time.
Apptainer may itself create transient runtime files in the generated directory;
the directory is a local runtime product and must not be committed.

The sandbox binds the host `/bin`, `/usr`, `/lib`, `/lib64`, `/etc`, `/cvmfs`,
`/sdf` and `/tmp`, plus the checkout. An additional bind exposes ROOT's library
directory at its baked-in LCG build path. This recreates the mechanism of the
original successful external wrapper; it does not isolate the host environment.

Optional environment overrides:

| Variable | Meaning |
|---|---|
| `AUDIT_REPO_ROOT` | Absolute checkout root; default derived from this script |
| `AUDIT_G4_ENV` | Absolute Geant4/ROOT setup script |
| `AUDIT_G4_ROOTFS` | Runtime sandbox path; default `packaging/rootfs` |
| `AUDIT_ROOT_LIB` | ROOT library path used for the special bind |
| `AUDIT_APPTAINER` | Apptainer executable, default `apptainer` |
| `AUDIT_PYTHON`, `AUDIT_UV` | Interpreter/runner overrides used by validation, capture, and the spin-injection launcher |
| `AUDIT_G4_RUNNER` | Wrapper override recognized by `primaries/run_spin.sbatch` |

These overrides configure paths. Different compiler, Geant4, ROOT or library
versions require independent validation; they are not established equivalents.

## Small validation and preparation order

`../photonsim_deep/primaries/run_spin.sbatch` builds the two primary-injection
probes and generates all fixtures it needs, including the input without the
optional spin branch. It now uses this packaged wrapper and the checkout root
from `SLURM_SUBMIT_DIR` or `AUDIT_REPO_ROOT`. Override `AUDIT_PYTHON` and
`AUDIT_UV` for a different interpreter/runner; the compiler/Geant4 build still
uses the exact CVMFS stack listed above. Its missing-branch test checks five GENIE entries and all
twenty expected particle PDGs, so default-electron fallback cannot pass it.

Once those probe executables are built, run from the checkout root:

```bash
sbatch audit_wc_20261006/packaging/validate_runtime.sbatch
```

This creates a new sandbox named for the job, regenerates the small primary
fixtures, repeats the spin-preservation/rotation and absent-branch assertions,
and records a compact completion marker. It does not repeat transported physics
distributions. It requires the two probe binaries built by the previous step;
those local executables are intentionally not published.

The fresh-sandbox/injection checks passed in milano job **39974427** on
`sdfmilan270`, exit `0:0` in 10 seconds. See `runtime-39974427.log`. All 256
polarized muons retained unit spin after the isolated repair, and the
missing-branch case contained five valid GENIE entries with twenty expected
primaries. This validates this wrapper on the audited host/CVMFS stack, not a
different host, software stack, or every relocated historical launcher.

Other launchers preserve historical absolute checkout and external-wrapper
paths. Before rerunning one from another location, replace those with the new
checkout and this wrapper, adapt the Python/Slurm account paths, and regenerate
its input macros rather than executing old `.mac`/`.commands` files containing
the original absolute filenames. Keep `PYTHONPATH` equal to the absolute checkout
root, `JAX_PLATFORMS=cpu`, and `PYTHONDONTWRITEBYTECODE=1`; use
`PYTHONNOUSERSITE=1` for the isolated pinned environment.

Important preparation dependencies when raw evidence is excluded:

| Target | Required preparation |
|---|---|
| Wrong-tree production reproduction; minimal Cerenkov macros | `photonsim_physics/run_audit.sbatch` builds `PhotonSim_v102` |
| Single-particle end-to-end integration | The same source audit produces electron/muon/pion/gamma ROOT fixtures |
| Bomb occupancy, boundary census, bright-pulse replay | `end_to_end/bomb.sbatch` produces the bomb ROOT and HDF5 outputs |
| Spin injection | `primaries/run_spin.sbatch` prepares both fixture kinds and builds its own probes |
| Cerenkov convergence/fine-step follow-ups | `photonsim_deep/cherenkov/run_tracks.sbatch` builds the dense-index executable |
| Geometry population follow-ups | `geometry/run.sbatch` produces `reflection_sample.npz`; some checks also need the pinned environment |
| Capture/EM analysis-only launchers | `transport/run_capture.sbatch` / `run_transport.sbatch` produce the CSV ledgers |

Use the retained compact JSON results for review. Raw ROOT/HDF5, photon/step
ledgers, large CSV/TSV/NPZ fixtures, compiled executables, caches and virtual
environments are regeneration products. A script that only analyzes such a file
requires its generating step first. Published results describe the original
recorded jobs; rerunning launchers may overwrite local result files.
