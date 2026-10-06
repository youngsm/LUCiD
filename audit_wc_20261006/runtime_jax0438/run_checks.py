"""Audit the current checkout with JAX 0.4.38, exclusively on allowed CPU nodes."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}
assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert os.environ.get("PYTHONNOUSERSITE") == "1"
import jax
import jaxlib
import flax
import optax
import numpy
import lucid.propagation.base as base

root = Path.cwd().resolve()
out = root / "audit_wc_20261006/runtime_jax0438"
assert Path(base.__file__).resolve() == root / "lucid/propagation/base.py", base.__file__
assert jax.__version__ == jaxlib.__version__ == "0.4.38"
assert jax.default_backend() == "cpu"
metadata = dict(host=socket.gethostname(), partition=os.environ["SLURM_JOB_PARTITION"],
                job_id=os.environ["SLURM_JOB_ID"], python=sys.version,
                interpreter=sys.executable, jax=jax.__version__, jaxlib=jaxlib.__version__,
                flax=flax.__version__, optax=optax.__version__, numpy=numpy.__version__,
                jax_source=jax.__file__, base_source=base.__file__,
                base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print(json.dumps(metadata), flush=True)
(out / "environment.json").write_text(json.dumps(metadata, indent=2))
os.environ["AUDIT_OUTPUT_DIR"] = str(out)
jobs = [
    ("reflection_current", ["audit_wc_20261006/geometry/repro_reflection_selfhit.py"]),
    ("reflection_fixed", ["audit_wc_20261006/geometry/repro_reflection_selfhit.py", "--fixed"]),
    ("transport", ["audit_wc_20261006/optical_transport/check_transport_physics.py"]),
]
statuses = {}
for name, args in jobs:
    print("RUN", name, flush=True)
    with (out / (name + ".log")).open("w") as log:
        result = subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    statuses[name] = result.returncode
    print("EXIT", name, result.returncode, flush=True)
    (out / "exit_codes.json").write_text(json.dumps(statuses, indent=2))
assert statuses["transport"] == 0, statuses
