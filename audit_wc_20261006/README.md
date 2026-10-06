# SK_WAND physics audit package

**Read [REPORT.md](REPORT.md)** for the single consolidated audit report. It covers LUCiD DATA mode, PhotonSim, particle-bomb/single-particle configurations, and GENIE inputs, with findings ordered by impact, minimal reproductions, candidate fixes, and explicit physical limits.

This branch preserves the exact audited LUCiD revision as its parent. It adds audit evidence and reproduction support; it does not apply the proposed production fixes.

## What is published

- The consolidated report, compact numerical results, selected event ledgers and logs.
- Test and analysis programs, original macros, Slurm launchers, and isolated candidate patches.
- The pinned PhotonSim source snapshot and Geant4 reference sources with their license and attribution.
- Runtime/version metadata and a configurable Geant4 execution wrapper.
- `FILE_MANIFEST.json`, recording file paths, sizes, SHA-256 hashes and whether each retained local artifact is included or omitted.

**Large files are intentionally not pushed.** Raw ROOT/HDF5 files and binary photon observers, large event/step tables, rebuildable executables, Python environments, caches and runtime sandboxes remain on SDF. The publication policy caps individual included files at **2 MiB** and additionally excludes raw ROOT/HDF5/observer files and Cerenkov per-photon/per-step tables regardless of size. Small numerical fixtures such as selected CSV/TSV/NPZ files may be included. The manifest states the actual disposition of each artifact.

Copied paper/figure assets and review-only GENIE source files are omitted; citations and pinned-source provenance remain. See [third-party notices](packaging/licenses/NOTICE.txt). Supporting lane notes are historical evidence, not separate final audit reports; `REPORT.md` is authoritative when an earlier note differs.

## Reproduce a finding

1. Read its section in [REPORT.md](REPORT.md), including whether the patch was executed or only proposed.
2. Follow [runtime and preparation instructions](packaging/RUNTIME.md). The audit used Geant4 11.3.0 and specific ROOT/Python installations; this package does not bundle those dependencies or claim universal portability.
3. Obtain a **milano or roma CPU allocation** before importing scientific packages, compiling, simulating or running tests. GPU work, if added, must use **turing**, with at most 10 GPUs shared across the audit. This audit used no GPUs.
4. Run the relevant fixture-generation launcher before an analysis that reads omitted raw files. Historical macros and launchers contain original SDF paths; regenerate macros and adapt paths when relocating the checkout.

The packaged wrapper has been checked with a fresh empty Apptainer sandbox and the small real-Geant4 GENIE polarization probes. This is a bounded runtime check, not a rerun of every physical distribution on a fresh machine. The polarization regression also checks that the absent-branch fixture really imported GENIE primaries, so silent fallback electrons cannot falsely pass it.

```bash
# From the repository root, on S3DF:
sbatch audit_wc_20261006/photonsim_deep/primaries/run_spin.sbatch
sbatch audit_wc_20261006/packaging/validate_runtime.sbatch
```

The second command requires the probe executables produced by the first; submit it after the first succeeds. Other regeneration dependencies are listed in the runtime guide. Reruns may overwrite local result files; the published compact results describe the original recorded audit jobs.

## Package maintenance

`packaging/build_manifest.py` inventories the local evidence, validates report links, and writes a reviewed staging list. Run it on an allowed CPU allocation through `packaging/manifest.sbatch`. Its generated staging list and runtime directories are not publication artifacts. Do not use an unrestricted force-add of the whole audit directory: the local directory contains gigabytes of intentionally omitted data.
