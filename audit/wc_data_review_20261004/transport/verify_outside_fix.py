"""Check one-line final-origin mask in an isolated module; no production edit."""
import os
assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
from pathlib import Path

source = Path('lucid/simulation/simulator.py').read_text()
before = "        mask = jnp.arange(n_rays) < photon_data['N']\n"
after = "        mask = (jnp.arange(n_rays) < photon_data['N']) & get_inside_detector_flag(final_origins)\n"
assert source.count(before) == 1
namespace = {'__name__': 'transport_outside_fix'}
exec(compile(source.replace(before, after), 'transport_outside_fix', 'exec'), namespace)
reproducer = Path('audit/wc_data_review_20261004/transport/reproduce_outside.py').read_text()
reproducer = reproducer.replace(
    'from lucid.simulation.simulator import setup_event_simulator',
    'setup_event_simulator = patched_setup')
reproducer = reproducer.replace("assert rows[1]['total_pe'] > 10_000", "assert rows[1]['total_pe'] == 0.0")
reproducer = reproducer.replace('transport/outside_results.json', 'transport/outside_fixed_results.json')
exec(compile(reproducer, 'outside_fix_reproducer', 'exec'),
     {'__name__': '__main__', 'patched_setup': namespace['setup_event_simulator']})
print('PASS: final-origin bounds mask removes all exterior PE.', flush=True)
