from pathlib import Path
import json
import numpy as np
from lucid.simulation.digitizer import digitize_event, resolve_model_config

HERE = Path(__file__).resolve().parent
physics = json.loads(Path("config/SK_WAND_physics_config.json").read_text())
inputs = (np.array([0,0]), np.array([0.,2000.]), np.array([1.,1.]), 1)
direct = digitize_event(*inputs, resolve_model_config(None))
expected = digitize_event(*inputs, resolve_model_config(physics["digitizer"]))
assert direct.digit_time.tolist() == [0.]
assert direct.digit_pe_true.tolist() == [2.]
assert expected.digit_time.tolist() == [0.,2000.]
assert expected.digit_pe_true.tolist() == [1.,1.]
results = {"direct_time": direct.digit_time.tolist(),
           "direct_charge": direct.digit_pe_true.tolist(),
           "configured_time": expected.digit_time.tolist(),
           "configured_charge": expected.digit_pe_true.tolist()}
(HERE/"cli_readout_results.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
