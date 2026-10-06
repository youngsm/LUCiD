"""Inventory the bounded publication package without moving or deleting evidence."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re

assert os.environ.get("SLURM_JOB_PARTITION") in ("milano", "roma")
audit = Path(__file__).resolve().parents[1]
repo = audit.parent
cap = 2 * 1024 * 1024
generated = {"FILE_MANIFEST.json", "packaging/staging-files.txt",
             "packaging/package-check.json", ".gitignore"}
environment_dirs = {"venv", "uv_cache", "__pycache__", ".ipynb_checkpoints"}


def reason(path, size, magic):
    rel = path.relative_to(audit).as_posix()
    if rel.startswith("digitizer/refs/") and path.suffix in {".pdf", ".svg", ".png", ".html"}:
        return "external publication: retain citation, not copied article/figure"
    if path.name.startswith("GENIE_R-"):
        return "review-only GENIE source: pinned upstream provenance is included"
    if magic == b"\x7fELF" or path.suffix in {".o", ".so", ".pyc"}:
        return "rebuildable executable or runtime artifact"
    if path.suffix in {".root", ".h5", ".hdf5", ".bin"}:
        return "raw binary evidence: retained locally or regenerated"
    if rel.startswith("photonsim_deep/cherenkov/") and path.suffix == ".csv":
        return "raw Cerenkov event/step/photon table: retained locally or regenerated"
    if size > cap:
        return "larger than the 2 MiB publication cap"
    return None


rows = []
included = []
ignore = ["# Generated publication exclusions; local evidence remains on SDF.",
          "venv/", "uv_cache/", "__pycache__/", "rootfs*/", "*.o", "*.pyc",
          "/packaging/staging-files.txt", "/packaging/package-check.json",
          "/packaging/manifest-*.log"]
for current, dirs, files in os.walk(audit, followlinks=False):
    dirs[:] = sorted(d for d in dirs if d not in environment_dirs
                     and not d.startswith("rootfs") and d != ".git")
    for name in sorted(files):
        path = Path(current) / name
        rel = path.relative_to(audit).as_posix()
        if rel in generated:
            continue
        if rel.startswith("packaging/manifest-") and path.suffix == ".log":
            continue
        if path.is_symlink():
            rows.append({"path": rel, "included": False,
                         "reason": "local convenience symlink", "target": os.readlink(path)})
            ignore.append("/" + rel)
            continue
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            magic = stream.read(4)
            digest.update(magic)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        why = reason(path, size, magic)
        row = {"path": rel, "bytes": size, "sha256": digest.hexdigest(),
               "included": why is None}
        if why:
            row["reason"] = why
            ignore.append("/" + rel)
        else:
            included.append(path.relative_to(repo).as_posix())
        rows.append(row)

manifest = {"schema": 1, "audit_base_commit": "20df3094bd563162d40c6ca34c5ad25e1e1d3656",
            "publication_cap_bytes": cap,
            "policy": "No large files; compact evidence and reproducible source only.",
            "local_evidence_root": str(audit),
            "environment_directories": "Excluded without enumerating installed dependencies/caches.",
            "self_reference": "Manifest and generated .gitignore are included but not self-hashed.",
            "files": rows}
(audit / "FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
(audit / ".gitignore").write_text("\n".join(ignore) + "\n")
included += ["audit_wc_20261006/FILE_MANIFEST.json", "audit_wc_20261006/.gitignore"]
selected = set(included)

report = (audit / "REPORT.md").read_text()
anchors = set()
for line in report.splitlines():
    if line.startswith("#"):
        heading = re.sub(r"^#+\s+", "", line).lower()
        anchors.add(re.sub(r"[^\w\- ]", "", heading).replace(" ", "-"))
broken = []
for target in re.findall(r"\]\(([^)]+)\)", report):
    if target.startswith(("https://", "http://")):
        continue
    if target.startswith("#"):
        if target[1:] not in anchors:
            broken.append(target)
    else:
        dest = audit / target
        if not dest.exists() or dest.relative_to(repo).as_posix() not in selected:
            broken.append(target)
assert not broken, f"Report links absent from publication: {broken}"
assert report.count("```") % 2 == 0

secrets = re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
for rel in included:
    p = repo / rel
    assert p.stat().st_size <= cap, rel
    assert not secrets.search(p.read_bytes()), f"Possible credential in {rel}"

(audit / "packaging/staging-files.txt").write_text("\n".join(sorted(included)) + "\n")
out = {"job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ["SLURM_JOB_PARTITION"],
       "included_files": len(included), "included_bytes": sum((repo / r).stat().st_size for r in included),
       "excluded_by_reason": dict(Counter(r.get("reason") for r in rows if not r["included"])),
       "report_links": "all local targets included; section anchors resolve",
       "credential_pattern_scan": "passed", "maximum_file_bytes": max((repo / r).stat().st_size for r in included)}
(audit / "packaging/package-check.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2), flush=True)
