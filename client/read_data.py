from torch.utils.data import Subset
import pandas as pd
import os
import torch
import numpy as np
import NmrDataset.NmrDataset
from torch.utils.data import TensorDataset, DataLoader

def read_data(data_path):
    offset_dict = {}
    with open(data_path, 'rb') as f:
        number_of_lines = len(f.readlines())
        for line in range(number_of_lines):
            offset = f.tell()
            offset_dict[line] = offset

    dataset = NmrDataset(data_path, offset_dict)
    return dataset