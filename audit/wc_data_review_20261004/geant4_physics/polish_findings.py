from pathlib import Path
p=Path('findings.md');s=p.read_text()
s=s.replace('All accepted numerical evidence comes from authorized milano CPU allocation `39894010` on `sdfmilan269`, with no GPUs.','All source simulations ran in authorized milano CPU allocation `39894010` on `sdfmilan269`, with no GPUs; final ROOT regressions, tail summaries and native material inspection ran in replacement milano allocation `39895962`.')
s=s.replace('All later source inspection and edits also ran inside allocation `39894010` after the user tightened the no-login-operations rule.','All source inspection and edits after the user tightened the no-login-operations rule ran inside these authorized milano allocations.')
start=s.index('## Scope\n');end=s.index('## Confirmed: missing thermal-neutron',start)
scope=s[start:end];s=s[:start]+s[end:]+'\n'+scope
p.write_text(s)
