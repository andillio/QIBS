import torch
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

N = 8000
x = np.random.randn(N,N)

if yt.is_root():

    start = time.time()
    x2 = x.dot(x)
    print(f"CPU matmul (sans allocation): {time.time()-start:6f} seconds")


start = time.time()
assert torch.cuda.is_available(), "GPU not working"
device = torch.device(f'cuda:{get_id()}')
x_gpu = torch.tensor(x).to(device)
x2_gpu = torch.matmul(x_gpu, x_gpu)
print(f"GPU {get_id()} (with migration): {time.time()-start:6f} seconds")


