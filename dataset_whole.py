import os

import numpy as np
from numpy import genfromtxt
import torch
from torch.utils.data import Dataset

from lib.misc import StandardScaler

class PeMSD7(Dataset):

    def __init__(self, root, subset, mean=0., std=0.):
        super(PeMSD7, self).__init__()
        self.subset = subset
        self.data = None
        self.file_idx = []
        self._read_from_folder(root, self.subset)
        self.mean = mean
        self.std = std
        self._normalize()

    def _read_from_folder(self, root, subset):
        if subset == 'test':
            root = os.path.join(root, 'test')
            all_data = []
            for file in os.listdir(root):
                assert os.path.isfile(os.path.join(root, file))
                data = genfromtxt(os.path.join(root, file), delimiter=',')  # (12, #station)
                all_data.append(data)
                self.file_idx.append(int(file.split('.')[0]))
            self.data = np.stack(all_data)  # (#file, 12, #station)
        else:  # train and val
            data = genfromtxt(os.path.join(root, 'V_228.csv'), delimiter=',')  # (44*288, #station)
            data = data.reshape((-1, 288, data.shape[1]))
            idx = []
            if subset == 'train':
                for i in range(data.shape[0]):
                    if i % 5 != 0:
                        idx.append(i)
            else:
                for i in range(data.shape[0]):
                    if i % 5 == 0:
                        idx.append(i)
            self.data = np.stack([data[i] for i in idx])

            # self.data = data

        num_time_intervals = 288
        time_gen = np.tile(np.linspace(0., 1., num=num_time_intervals, endpoint=False),
                           (self.data.shape[2], 1)).T  # (288, #station)
        self.time_gen = torch.tensor(time_gen[:12], dtype=torch.float32)
        self.time_gen_scale = torch.tensor(time_gen[:24:2], dtype=torch.float32)

    def _normalize(self):
        if self.mean == 0.:
            self.mean = self.data.mean()
        if self.std == 0.:
            self.std = self.data.std()
        self.scaler = StandardScaler(mean=self.mean, std=self.std)
        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        if self.subset != 'test':
            return self.data.shape[0] * (self.data.shape[1] - 23)
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        # get (12, #station, 2)
        if self.subset != 'test':
            sample_per_file = self.data.shape[1] - 23
            ix, iy = int(idx / sample_per_file), int(idx % sample_per_file)
            x = self.data[ix, iy : iy + 12]
            y = self.data[ix, iy + 12 : iy + 24]
            x = torch.stack((x, self.time_gen), dim=2)
            y = torch.stack((y, self.time_gen), dim=2)
            return x, y
        else:
            x = self.data[idx]
            x = torch.stack((x, self.time_gen), dim=2)
            return x, self.file_idx[idx]

if __name__ == '__main__':
    dataset = PeMSD7('data/PEMS-D7', 'val')
    print(len(dataset))
