from .. import (
    np, torch
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed);