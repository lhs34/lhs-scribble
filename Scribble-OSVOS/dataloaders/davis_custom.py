from __future__ import division

from mypath import Path
import os
import numpy as np
import torch, cv2

from torch.utils.data import Dataset
import json
from PIL import Image

class DAVISCUSTOM2017(Dataset):
    """Custom DAVIS 2017 dataset constructed using the PyTorch built-in functionalities"""
    def __init__(self, split='val',
                       root=Path.db_root_dir(),num_frames=None,
                       custom_frames=None,transform=None,
                       retname=False,
                       seq_name=None,
                       obj_id=None,
                       gt_only_first_frame=False,
                       no_gt=False,
                       batch_gt=False,
                       rgb=False,
                       effective_batch=None):
        return 
    
    def __get_item__(self):
        pass

    def __len__(self):
        pass