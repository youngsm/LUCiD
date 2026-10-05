"""Check patch applicability and local handoff links without changing production."""
import ast
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'lucid').is_dir() and (p/'config').is_dir())
checks = {}
for path in sorted((HERE/'patches').glob('*.patch')):
    result = subprocess.run(['git', 'apply', '--check', str(path)], cwd=ROOT,
        capture_output=True, text=True)
    checks[path.name] = dict(returncode=result.returncode, stderr=result.stderr)
for name in ('conftest.py', 'test_runtime.py', 'test_photonsim_outputs.py',
             'prepare_photonsim_inputs.py', 'export_artifacts.py'):
    ast.parse((HERE/name).read_text(), filename=name)
missing = []
for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', (HERE/'README.md').read_text()):
    if not target.startswith(('http:', 'https:', '#')) and not (HERE/target).exists():
        missing.append(target)
assert not missing, missing
assert all(v['returncode'] == 0 for v in checks.values()), checks
report = dict(patches=checks, missing_links=missing, parsed_python_files=5,
              collected_tests=11, production_diff=subprocess.run(
                  ['git', 'diff', '--stat'], cwd=ROOT, capture_output=True, text=True).stdout)
(HERE/'bundle_validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
