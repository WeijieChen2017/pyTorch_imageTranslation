from torch.utils.data import Dataset

import os
import numpy as np
from glob import glob

class DatasetFromFolder(Dataset):
    def __init__(self, data_dir_X, data_dir_Y, batch_size, shuffle=False):
        super(DatasetFromFolder, self).__init__()
        self.filenames_X = sorted(glob(os.path.join(data_dir_X,'*.npy'),recursive=True))
        self.filenames_Y = sorted(glob(os.path.join(data_dir_Y,'*.npy'),recursive=True))
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            temp = list(zip(self.filenames_X, self.filenames_Y))
            random.shuffle(temp)
            self.filenames_X, self.filenames_Y = zip(*temp)

def __getitem__(self, index):

    batch_x_fns = self.filenames_X[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y_fns = self.filenames_Y[idx * self.batch_size:(idx + 1) * self.batch_size]

    batch_x = np.array( [ np.load(fn) for fn in batch_x_fns ] )
    batch_y = np.array( [ np.load(fn) for fn in batch_y_fns ] )

    return batch_x, batch_y

def __len__(self):
    return len(self.filenames_X // self.batch_size)
