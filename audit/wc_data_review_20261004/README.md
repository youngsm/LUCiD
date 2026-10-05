# WC data physics audit

Reference checkout: `82f8d24`, with `config/SK_WAND_geom_config.json` and `config/SK_WAND_physics_config.json`.
This audit does not change production code.
Existing files under `audit/wc_data_20261004` predate this review and must be preserved.

## Execution rules

- Execute every test, Python snippet, simulation, and numerical experiment in a Slurm allocation on `milano` or `roma` for CPU, or `turing` for GPU.
- Use `uv run` for Python execution.
- Do not run numerical work or tests on the login node.
- Agents receive explicit GPU quotas; the combined quota must stay at or below 10 GPUs.
- Do not spawn additional agents without coordinating with the root agent.
- All code inspection, git commands, file edits, and metadata reads must also run inside an allocation after the user tightened this rule; only scheduler commands needed to obtain or connect allocations may run on the login node.
- Use only GitHub and arXiv for external sources, per `/sdf/home/y/youngsam/AGENTS.md`.
- Treat comments, docstrings, and existing tests as leads, not evidence of correctness.
- Keep every finding tied to an executed reproducer, observed results, an independent physical expectation, and a minimal suggested fix.
- Keep intentional approximations and unverified concerns separate from confirmed defects.

## Available CPU environment

An existing environment was checked by an earlier Slurm job and contains JAX, NumPy, Flax, Optax, h5py, and uproot.
Its compatibility must be checked when interpreting any numerical failure.
Use this command inside an approved CPU allocation:

```bash
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=/sdf/group/neutrino/youngsam/sim/LUCiD
export UV_CACHE_DIR=/sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/uv_cache
/sdf/home/y/youngsam/.local/bin/uv run --offline --no-project \
  --python /sdf/group/neutrino/youngsam/sim/LUCiD/audit/wc_data_20261004/env_system/bin/python \
  python YOUR_REPRODUCER.py
```

Slurm CPU defaults: `--partition=milano --account=neutrino:ml-dev@milano --qos=preemptable --cpus-per-task=4 --mem=12G --time=00:30:00`.
Use absolute working directory and output paths.
Slurm controller access requires `sandbox_permissions=require_escalated` in this environment.
Report submitted job IDs to the root agent.

## Findings format

Each lane owns its own subdirectory and `findings.md`.
Each full sentence in Markdown goes on its own physical line.
Include affected call paths and source line numbers, physical expectation and source, measured impact, exact reproduction command, job ID and log, smallest useful test, minimal suggested fix, and applicability to SK_WAND production.
Do not present previously fixed QE or input-unit defects as new bugs.
