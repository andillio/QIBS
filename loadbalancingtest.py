import numpy as np
import yt; yt.enable_parallelism()
import time

def get_id():
    """
    This function returns True if it is on the root processor of the
    topcomm and False otherwise.
    """
    from yt.config import ytcfg
    cfg_option = "__topcomm_parallel_rank"
    return ytcfg.getint("yt", cfg_option)

if yt.is_root():
    print('starting')

id = get_id()
print(id)

times = [10, *np.ones(9)]

n = 0
for t in yt.parallel_objects(times,0,dynamic=False):

    n += 1

    print(f"Process {id} doing {t}, current is {n}th") 

    time.sleep(t)

print('next batch')

n = 0
for t in yt.parallel_objects(times,0, dynamic=True):

    n += 1

    print(f"Process {id} doing {t}, current is {n}th") 

    time.sleep(t)