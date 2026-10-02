import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets deterministic seeds across Python random, NumPy, and PyTorch (CPU and CUDA).
    Ensures CuDNN determinism if running on GPU.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Random seed set to {seed} (CUDA deterministic: {torch.cuda.is_available()})")
