"""Measure escaped photon-source fractions after production-equivalent placements."""
import json
import os
from pathlib import Path
import numpy as np
import uproot
from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.sources.event_generation import _random_rotation_matrix
from lucid.sources.root_reader import _read_event_raw
from lucid.sources.writer import sample_translation_vector

BASE = Path(__file__).resolve().parent


def main():
    assert os.environ.get('SLURM_JOB_PARTITION') in ('milano', 'roma')
    geometry = DetectorGeometry.from_config('config/SK_WAND_geom_config.json', temperature=0.0, deposit_leg_bound=True)
    detector = geometry.detector
    bounds = {'type': 'cylinder', 'radius': float(detector.r), 'height': float(detector.H)}
    rng = np.random.default_rng(15973)
    summary = {'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'partition': os.environ['SLURM_JOB_PARTITION'], 'bounds': bounds, 'production_vertex_r_z_fraction': .9, 'samples': []}
    for sample in ('mu1000', 'mu3000', 'e100'):
        path = BASE.parent / 'geant4_physics' / f'{sample}.root'
        with uproot.open(path) as f:
            entries = f['OpticalPhotons'].num_entries
        photon_counts, outside_counts, event_fractions = [], [], []
        max_extent = 0.
        worst_fraction = -1.
        worst = None
        for event in range(min(entries, 20)):
            raw = _read_event_raw(str(path), event)
            source_origins = np.asarray(raw['photon_origins'], np.float64)
            if len(source_origins) == 0:
                continue
            max_extent = max(max_extent, float(np.max(np.linalg.norm(source_origins, axis=1))))
            for placement in range(25):
                R = _random_rotation_matrix(rng)
                vertex = sample_translation_vector(bounds, rng).astype(np.float64)
                origins = source_origins @ R.T + vertex
                inside = (np.sum(origins[:, :2]**2, axis=1) < detector.r**2) & (np.abs(origins[:, 2]) < detector.H/2)
                outside = int((~inside).sum())
                photon_counts.append(len(origins)); outside_counts.append(outside)
                event_fractions.append(outside / len(origins))
                if outside / len(origins) > worst_fraction:
                    worst_fraction = outside / len(origins)
                    worst = {
                        'event': event, 'placement': placement, 'rotation': R.copy(),
                        'vertex_m': vertex.copy(), 'origins_m': origins.astype(np.float32),
                        'directions': (np.asarray(raw['photon_directions'], np.float64) @ R.T).astype(np.float32),
                        'times_ns': np.asarray(raw['photon_times']),
                        'wavelengths_nm': np.asarray(raw['photon_wavelengths']),
                        'segment_ids': np.asarray(raw['photon_segment_index_raw']),
                        'outside_mask': ~inside,
                    }
        row = {'sample': sample, 'source_events': min(entries, 20), 'placements_per_event': 25, 'evaluated_placements': len(event_fractions), 'total_photon_placements': sum(photon_counts), 'outside_photon_placements': sum(outside_counts), 'outside_photon_fraction': sum(outside_counts)/sum(photon_counts), 'fraction_placements_with_outside_photons': float(np.mean(np.array(event_fractions) > 0)), 'maximum_outside_fraction_in_one_placement': float(max(event_fractions)), 'max_origin_distance_from_vertex_m': max_extent}
        row['worst_placement'] = {'event': int(worst['event']), 'placement': int(worst['placement']), 'vertex_m': worst['vertex_m'].tolist(), 'rotation': worst['rotation'].tolist()}
        np.savez_compressed(BASE / f'{sample}_worst_placement.npz', **worst)
        summary['samples'].append(row)
        print(json.dumps(row), flush=True)
    (BASE / 'outside_population_results.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
