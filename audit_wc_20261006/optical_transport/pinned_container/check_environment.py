"""Assert imports are the bind-mounted current checkout, then run CPU checks."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

assert os.environ.get("SLURM_JOB_PARTITION") in {"milano", "roma"}
assert os.environ.get("PYTHONNOUSERSITE") == "1"
assert os.environ.get("JAX_PLATFORMS") == "cpu"
import jax
import lucid
import lucid.propagation.base as base

root = Path("/opt/LUCiD")
out = root / "audit_wc_20261006/optical_transport/pinned_container"
assert Path(base.__file__).resolve() == root / "lucid/propagation/base.py", base.__file__
assert jax.default_backend() == "cpu"
metadata = dict(host=socket.gethostname(), partition=os.environ["SLURM_JOB_PARTITION"],
                job_id=os.environ["SLURM_JOB_ID"], python=sys.version,
                interpreter=sys.executable, jax=jax.__version__, jax_source=jax.__file__,
                lucid_source=lucid.__file__, base_source=base.__file__,
                base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print(json.dumps(metadata), flush=True)
(out / "environment.json").write_text(json.dumps(metadata, indent=2))
os.environ["AUDIT_OUTPUT_DIR"] = str(out)
jobs = [
    ("transport", ["audit_wc_20261006/optical_transport/check_transport_physics.py"]),
    ("reflection_current", ["audit_wc_20261006/geometry/repro_reflection_selfhit.py"]),
    ("reflection_fixed", ["audit_wc_20261006/geometry/repro_reflection_selfhit.py", "--fixed"]),
    ("full_propagator", [str(out / "check_full_propagator.py")]),
    ("fused_reflections", [str(out / "check_fused_reflections.py")]),
]
statuses = {}
for name, args in jobs:
    print("RUN", name, flush=True)
    with (out / (name + ".log")).open("w") as log:
        completed = subprocess.run([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT,
                                   check=False)
    statuses[name] = completed.returncode
    print("EXIT", name, completed.returncode, flush=True)
(out / "exit_codes.json").write_text(json.dumps(statuses, indent=2))
# A version-dependent deterministic reproduction may fail its assertion; preserve
# the full diagnostics instead of stopping before the fixed/full-map controls.
assert statuses["transport"] == 0, statuses
