import json
import numpy as np
from test_neutron_capture import capture_times_us
out={}
for name in ('thermal_neutron_fixed.root','thermal_neutron_thermalelastic.root'):
    import os
    os.environ['PHOTONSIM_THERMAL_ROOT']=name
    t=capture_times_us.__wrapped__()
    out[name]={'n':len(t),'fraction_over_1ms':float(np.mean(t>1000)),'fraction_over_2ms':float(np.mean(t>2000)),'percentile95_us':float(np.quantile(t,.95)),'max_capture_time_us':float(np.max(t))}
print(json.dumps(out,indent=2))
open('thermal_tail_summary.json','w').write(json.dumps(out,indent=2)+'\n')
