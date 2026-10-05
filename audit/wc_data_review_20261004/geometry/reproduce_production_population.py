"""The standard production vertex volume; ray directions are isotropic probes."""
import json
import os
from pathlib import Path
import numpy as np
from lucid.geometry.detector_geometry import DetectorGeometry
from reproduce_geometry import analyze
assert os.environ['SLURM_JOB_PARTITION'] in ('milano','roma')
g=DetectorGeometry.from_config('config/SK_WAND_geom_config.json',temperature=0.0,deposit_leg_bound=True)
rng=np.random.default_rng(1701042026)
n=20000
u=rng.uniform(size=(n,3))
r=.9*g.detector.r*np.sqrt(u[:,0]); theta=2*np.pi*u[:,1]
o=np.column_stack([r*np.cos(theta),r*np.sin(theta),.9*g.detector.H*(u[:,2]-.5)])
d=rng.normal(size=(n,3))
row,examples=analyze(g,o,d,'standard_production_vertex_volume_isotropic')
cap=np.broadcast_to(np.array([0.,0.,.9*g.detector.H/2]),(n,3)).copy()
row_cap,examples_cap=analyze(g,cap,rng.normal(size=(n,3)),'standard_production_top_vertex_isotropic')
Path('audit/wc_data_review_20261004/geometry/production_population_results.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],metrics=[row,row_cap],examples=[examples,examples_cap]),indent=2)+'\n')
