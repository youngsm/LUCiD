"""Record the inherited audit Python environment on an allowed CPU allocation."""
import os
assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
names = sorted({dist.metadata["Name"] for dist in metadata.distributions()
                if dist.metadata.get("Name")}, key=str.lower)
# Resolve by name through the active interpreter search path, retaining the
# overlay's selected version when inherited metadata has a second version.
versions = {name: metadata.version(name) for name in names}
(base / "requirements-baseline-resolved.txt").write_text(
    "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
modules = {}
for name in ("jax", "jaxlib", "flax", "optax", "numpy", "scipy", "h5py", "uproot", "awkward"):
    mod = importlib.import_module(name)
    modules[name] = {"version": getattr(mod, "__version__", metadata.version(name)),
                     "source": str(Path(mod.__file__).resolve())}
record = {"job": os.environ.get("SLURM_JOB_ID"),
          "partition": os.environ["SLURM_JOB_PARTITION"],
          "python": sys.version, "executable": sys.executable,
          "modules": modules,
          "note": "Visible package metadata including inherited system sites; not a minimal or portable environment lock."}
(base / "environment-baseline.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
