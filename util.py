import os
import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    #cudnn 시드 고정 : true, false 
    #cudnn 시드 랜덤 : false, true
    # https://www.facebook.com/groups/PyTorchKR/permalink/1010080022465012/
    # https://hoya012.github.io/blog/reproducible_pytorch/
    
    torch.backends.cudnn.deterministic = True ## cuda.manual_seed 안에 포함된다고 함. 하지만, 찜찜해서 또 넣음
    torch.backends.cudnn.benchmark = False