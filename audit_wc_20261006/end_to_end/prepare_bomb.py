"""Generate ten ordinary events from the exact main WAND bomb config."""
from pathlib import Path
import json
import os

assert os.environ.get('SLURM_JOB_PARTITION') in {'milano', 'roma'}
from lucid.production.generate_macro import generate_macro

base = Path(__file__).resolve().parent
source = base / 'bomb_source'
source.mkdir(exist_ok=True)
config_path = base.parents[1] / 'lucid/production/configs/GeV/01_pbomb.json'
original_text = config_path.read_text()
(source / 'original_config.json').write_text(original_text)
config = json.loads(original_text)
(source / 'bomb.mac').write_text(generate_macro(
    config, str(source / 'bomb.root'), 10, photonsim_seeds=(314159, 271828)))
print('Generated ten-event macro from unchanged', config_path, flush=True)
