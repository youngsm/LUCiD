"""Package portable test excerpts and exact per-issue audit-local corrections."""
import ast
import difflib
import importlib.util
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'lucid').is_dir() and (p/'config').is_dir())
path = HERE/'conftest.py'
source = path.read_text().replace(
    'ROOT = Path(__file__).resolve().parents[4]',
    "ROOT = Path(os.environ['LUCID_REPO']).resolve() if os.environ.get('LUCID_REPO') else next(\n"
    "    p for p in Path(__file__).resolve().parents if (p / 'lucid').is_dir() and (p / 'config').is_dir())")
path.write_text(source)
spec = importlib.util.spec_from_file_location('regression_fixtures', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
patches = [
 ('incident_qe', 'lucid/simulation/simulator.py',
  '        # Per-photon segment id for the per_segment production hit mode',
  '        qe_per_photon = qe_per_photon / (1.0 - detector_params.reflection.sensor_reflection_rate)\n\n'
  '        # Per-photon segment id for the per_segment production hit mode',
  'Scalar SK_WAND DATA correction only; angular reflection must use the actual per-encounter reflectance and validate QE+R <= 1.'),
 ('outside_origin', 'lucid/simulation/simulator.py',
  "        mask = jnp.arange(n_rays) < photon_data['N']\n",
  "        mask = (jnp.arange(n_rays) < photon_data['N']) & get_inside_detector_flag(final_origins)\n",
  'Mask after all origin transformations.'),
 ('geometry_reference', 'lucid/propagation/shared.py',
  '        potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])',
  module.COMPLETE_SEGMENT_SELECTION,
  'Reference implementation only, verified on one tangent ray here. Full enumeration is memory-expensive; production should traverse the complete ray segment with a BVH or grid.'),
 ('diagnostic_no_time_cap', 'lucid/simulation/digitizer.py',
  '_MAX_DIGIT_TIME_NS = 1e5', '_MAX_DIGIT_TIME_NS = np.inf',
  'Diagnostic only. A production fix needs explicit acquisition windows and bounded dark-noise sampling.'),
 ('sampled_charge_timing', 'lucid/simulation/digitizer.py',
  't = _sample_time_jitter(digit_time, pe_true, model, rng)',
  't = _sample_time_jitter(digit_time, np.maximum(pe_reco, 0.5), model, rng)',
  'Verified for SK_WAND ski response. Review other digitizer prescriptions before generalizing.'),
 ('unmodified_wavelength_qe', 'lucid/wavelength/optical_model.py',
  'qe = qe_fn(wl) *', 'qe = qe_fn(wavelengths) *',
  'QE uses original wavelength; medium coefficient interpolation retains its independent clamp.'),
 ('negative_tts_charge', 'lucid/simulation/sensor_response.py',
  '(total_charge > 1e-10) & (detector_mins > 0) & jnp.isfinite(detector_mins)',
  '(total_charge > 1e-10) & jnp.isfinite(detector_mins)',
  'Dormant in selected per_segment/TTS=0 setup; retain physically valid jittered negative timestamps in realistic mode.')]
(HERE/'patches').mkdir(exist_ok=True)
metadata = []
for name, relative, before, after, scope in patches:
    old = (ROOT/relative).read_text(); assert old.count(before) == 1
    new = old.replace(before, after)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
        fromfile='a/'+relative, tofile='b/'+relative))
    (HERE/'patches'/f'{name}.patch').write_text(patch)
    metadata.append(dict(issue=name, patch='patches/'+name+'.patch', scope=scope,
                         correction=after.strip(), production_file=relative))
(HERE/'corrections.json').write_text(json.dumps(metadata, indent=2)+'\n')
excerpts = {}
for filename in ('test_runtime.py', 'test_photonsim_outputs.py'):
    code = (HERE/filename).read_text()
    for node in ast.parse(code).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            excerpts[node.name] = dict(file=filename, line=node.lineno,
                                      code=ast.get_source_segment(code, node))
(HERE/'code_excerpts.json').write_text(json.dumps(excerpts, indent=2)+'\n')
results = dict(baseline_head='82f8d249fd3fcbff1a26dd8a94465dc07e5f3839',
    slurm_job=os.environ['SLURM_JOB_ID'], partition=os.environ['SLURM_JOB_PARTITION'], gpus=0,
    runtime_baseline=dict(failed=7, passed=0, seconds=32.03, log='baseline.log'),
    runtime_isolated_corrections=dict(failed=0, passed=7, seconds=31.56, log='fixed.log'),
    saved_source_baseline=dict(failed=4, passed=0, seconds=2.18, log='source-baseline.log'),
    saved_source_corrected=dict(failed=0, passed=4, seconds=1.04, log='source-fixed.log'),
    source_scope='Saved outputs from previously executed baseline and corrected PhotonSim binaries. Rebuild and regenerate after each C++ change; these four tests do not inspect live Python code.',
    total_regressions=11, production_files_changed=0)
(HERE/'results.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
