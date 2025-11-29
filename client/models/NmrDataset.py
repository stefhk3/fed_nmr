from torch.utils.data import Dataset
import numpy as np

class NmrDataset(Dataset):
    def __init__(self, large_file_path, offset_dict):
        self.large_file_path = large_file_path
        self.offset_dict = offset_dict
    
    def __len__(self):
        return len(self.offset_dict)
    
    def __getitem__(self, line):
        offset = self.offset_dict[line]
        with open(self.large_file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            linearray = np.fromstring(line, dtype=float, sep=',')
            X = linearray[:-1]
            y = linearray[-1:]

            X = X.astype('float32')
            y = y.astype('float32')

            X = np.expand_dims(X, 1)
            X /= 255

            return X,y