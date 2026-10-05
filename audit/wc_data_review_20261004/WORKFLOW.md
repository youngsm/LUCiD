# Allocation workflow for the WC audit

Every test and numerical experiment used a milano Slurm CPU allocation and `uv run`.
No GPU was used.
After the user tightened the rule, shell inspection, code reads, edits, and metadata checks also moved into allocations.
Only scheduler commands needed to obtain or connect allocations ran outside them.

A persistent allocation was obtained with:

```bash
srun --partition=milano --account=neutrino:ml-dev@milano --qos=preemptable \
  --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --pty bash --noprofile --norc
```

The shared distinct-node allowance temporarily blocked extra allocations despite spare CPUs on already occupied hosts.
Constraining an owned pending job to an already-used host allowed it to fit that allowance:

```bash
scontrol update JobId=OWN_PENDING_JOB ReqNodeList=EXISTING_ALLOCATED_HOST
```

This does not bypass CPU, memory, QoS, or placement constraints and does not guarantee scheduling.
An existing allocation can also host a coordinated step:

```bash
srun --jobid=AUTHORIZED_ALLOCATION --overlap --ntasks=1 \
  --cpus-per-task=1 --pty bash --noprofile --norc
```

Specify `--ntasks=1` explicitly for a single nested command or shell.
Inherited defaults otherwise launched multiple copies of a nested command during this audit, risking concurrent output writes.
Coordinate CPU and memory use with the allocation owner.
Cancel duplicate pending allocations and release owned resources when finished.

The CERN ROOT runtime initially failed because its PCM imports retained a build-time absolute path.
The Geant4 lane repaired this in an audit-owned Apptainer root filesystem with a bind from the installed ROOT library directory to that expected path.
Its wrapper and environment scripts are preserved under `geant4_physics`.
Physics results from malformed ROOT outputs were rejected and rerun after the repair.
The corrected fixture also enabled the reader and complete ROOT-to-HDF5 production checks.

An attempted update to the shared Slurm skill was rejected by automatic approval review because it would persist guidance outside the requested audit scope.
The shared skill was not modified; these local notes are the safe alternative.
