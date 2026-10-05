from pathlib import Path
import json
import re

base = Path(__file__).resolve().parent
reports = [base / 'REPORT.md', base / 'WORKFLOW.md', *sorted(base.glob('*/findings.md'))]
missing = []
for report in reports:
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if target.startswith(('http:', 'https:', '#')):
            continue
        candidate = report.parent / target.split('#', 1)[0]
        if not candidate.exists():
            missing.append({'report': str(report.relative_to(base)), 'target': target})
summary = {'report_files': len(reports), 'missing_local_links': missing,
           'main_report_lines': len((base / 'REPORT.md').read_text().splitlines())}
print(json.dumps(summary, indent=2))
(base / 'artifact_check.json').write_text(json.dumps(summary, indent=2) + '\n')
assert not missing
