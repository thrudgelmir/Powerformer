from transformers import set_seed
import torch, numpy as np

seeds = [0,42,777]

def seedval(seed_val=0) :
    set_seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
