from pathlib import Path
import h5py
import numpy as np
root = Path('audit/wc_data_review_20261004/end_to_end/writer_output')
for file in sorted(root.rglob('*.h5')):
    print(file.name)
    with h5py.File(file) as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name, obj.shape, obj.dtype, np.asarray(obj).ravel()[:8])
        f.visititems(visit)
