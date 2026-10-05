"""Live production regressions; opt-in fixes are audit-owned copies only."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

ROOT = Path(os.environ['LUCID_REPO']).resolve() if os.environ.get('LUCID_REPO') else next(
    p for p in Path(__file__).resolve().parents if (p / 'lucid').is_dir() and (p / 'config').is_dir())
HERE = Path(__file__).resolve().parent
GEOM = str(ROOT / 'config/SK_WAND_geom_config.json')
PHYS = str(ROOT / 'config/SK_WAND_physics_config.json')


def pytest_addoption(parser):
    parser.addoption('--isolated-fixes', action='store_true',
                     help='Exercise audit-owned corrected copies; never edit production.')




def isolated_module(name, relative_path, replacements):
    source = (ROOT / relative_path).read_text()
    for before, after in replacements:
        assert source.count(before) == 1, (relative_path, before)
        source = source.replace(before, after)
    path = HERE / f'{name}.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f'lucid_audit_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def api(request):
    import lucid.simulation.simulator as simulator
    import lucid.simulation.sensor_response as response
    import lucid.simulation.digitizer as digitizer
    import lucid.wavelength.optical_model as optical
    from lucid.geometry.detector_geometry import DetectorGeometry
    from lucid.detector_params import load_physics_config
    import jax.numpy as jnp

    if request.config.getoption('--isolated-fixes'):
        simulator = isolated_module('simulator_fixed', 'lucid/simulation/simulator.py', [
            ("        mask = jnp.arange(n_rays) < photon_data['N']\n",
             "        mask = (jnp.arange(n_rays) < photon_data['N']) & get_inside_detector_flag(final_origins)\n"),
            ('        # Per-photon segment id for the per_segment production hit mode',
             '        qe_per_photon = qe_per_photon / (1.0 - detector_params.reflection.sensor_reflection_rate)\n\n'
             '        # Per-photon segment id for the per_segment production hit mode')])
        digitizer = isolated_module('digitizer_fixed', 'lucid/simulation/digitizer.py', [
            ('_MAX_DIGIT_TIME_NS = 1e5', '_MAX_DIGIT_TIME_NS = np.inf'),
            ('t = _sample_time_jitter(digit_time, pe_true, model, rng)',
             't = _sample_time_jitter(digit_time, np.maximum(pe_reco, 0.5), model, rng)')])
        optical = isolated_module('optical_fixed', 'lucid/wavelength/optical_model.py', [
            ('qe = qe_fn(wl) *', 'qe = qe_fn(wavelengths) *')])
        response = isolated_module('response_fixed', 'lucid/simulation/sensor_response.py', [
            ('(total_charge > 1e-10) & (detector_mins > 0) & jnp.isfinite(detector_mins)',
             '(total_charge > 1e-10) & jnp.isfinite(detector_mins)')])

    geometry = DetectorGeometry.from_config(GEOM, temperature=0., deposit_leg_bound=True)
    if request.config.getoption('--isolated-fixes'):
        shared = isolated_module('shared_fixed', 'lucid/propagation/shared.py', [
            ('        potential_sensors = jax.lax.stop_gradient(inverted_sensor_map[idx])',
             COMPLETE_SEGMENT_SELECTION)])
        # The quadratic full search is confined to the one-ray geometry regression.
        geometry = geometry._replace(propagator=shared.create_propagator(
            geometry.detector, geometry.sensor_points, geometry.sensor_radius,
            temperature=0., deposit_leg_bound=True))
    dp, material, qe_path = load_physics_config(PHYS, num_sensors=geometry.num_sensors)
    model = digitizer.resolve_model_config(json.loads(Path(PHYS).read_text())['digitizer'])
    return SimpleNamespace(simulator=simulator, response=response, digitizer=digitizer,
        optical=optical, geometry=geometry, dp=dp, material=material, qe_path=qe_path,
        model=model, jnp=jnp)

# Executable DATA-only baseline algorithm: enumerate all PMT sphere entries,
# discard intersections beyond the water boundary, then keep the first PMT.
# A production implementation should traverse a BVH/grid to bound memory.
COMPLETE_SEGMENT_SELECTION = '''        d = photon_directions / jnp.linalg.norm(photon_directions, axis=1, keepdims=True)
        delta = sensor_positions[None, :, :] - photon_origins[:, None, :]
        along = jnp.sum(delta * d[:, None, :], axis=-1)
        perpendicular = delta - along[:, :, None] * d[:, None, :]
        disc = sensor_radius**2 - jnp.sum(perpendicular**2, axis=-1)
        entry = along - jnp.sqrt(jnp.maximum(disc, 0.0))
        valid = (disc > 0.0) & (entry > 0.0) & (entry <= t_geometry[:, None])
        bounded = jnp.where(valid, entry, jnp.inf)
        first = jnp.argmin(bounded, axis=1)
        first = jnp.where(jnp.isfinite(jnp.min(bounded, axis=1)), first, -1)
        potential_sensors = jax.lax.stop_gradient(first[:, None])'''


@pytest.fixture(scope='session')
def transparent_parameters(api):
    dp, jnp = api.dp, api.jnp
    return dp._replace(
        scattering=dp.scattering._replace(scatter_length=jnp.asarray(1e20),
                                         mie_scatter_length=jnp.asarray(1e20)),
        absorption=dp.absorption._replace(absorption_length=jnp.asarray(1e20)),
        reflection=dp.reflection._replace(wall_reflection_rate=jnp.asarray(0.)),
        response=dp.response._replace(tts=jnp.asarray(0.)))


@pytest.fixture(scope='session')
def transparent_sim(api, transparent_parameters):
    return api.simulator.setup_event_simulator(GEOM, 0, K=1, is_data=True,
        temperature=0., physics_config=PHYS, default_detector_params=transparent_parameters,
        wavelength_mode=False, hit_mode='per_segment', deposit_leg_bound=True)
