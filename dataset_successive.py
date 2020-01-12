import os

import numpy as np
from numpy import genfromtxt
import torch
from torch.utils.data import Dataset

from lib.misc import StandardScaler

class PeMSD7(Dataset):

    def __init__(self, root, subset, mean=0., std=0., seq_len=12, horizon=12):
        super(PeMSD7, self).__init__()
        self.subset = subset
        self.mean = mean
        self.std = std
        self.seq_len = seq_len
        self.horizon = horizon

        self.data = None
        self.file_idx = []
        self._read_from_folder(root, self.subset)
        self._normalize()


    def _read_from_folder(self, root, subset):
        if subset == 'test':
            root = os.path.join(root, 'test')
        else:  # train and val
            root = os.path.join(root, 'train')
        all_data = []
        for file in os.listdir(root):
            assert os.path.isfile(os.path.join(root, file))
            if subset == 'train':
                if int(file.split('.')[0]) % 5 == 0:  # used for validation
                    continue
            elif subset == 'val':
                if int(file.split('.')[0]) % 5 != 0:
                    continue
            data = genfromtxt(os.path.join(root, file), delimiter=',')  # (288, #station) / (12, #station)
            all_data.append(data)
            self.file_idx.append(int(file.split('.')[0]))
        # self.data = np.stack(all_data)  # (#file, 288, #station) / (#file, 12, #station)
        self.data = np.concatenate(all_data)  # (#file * 288, #station) / (#file * 12, #station)

        num_time_intervals = 288
        time_gen = np.tile(np.linspace(0., 1., num=num_time_intervals, endpoint=False),
                           (self.data.shape[1], 1)).T  # (288, #station)
        self.time_gen = torch.tensor(time_gen[:self.seq_len], dtype=torch.float32)
        self.time_gen_y = torch.tensor(time_gen[:self.horizon], dtype=torch.float32)  # NOTE: not used in fact
        # self.time_gen_scale = torch.tensor(time_gen[:24:2], dtype=torch.float32)

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
            return self.data.shape[0] - (self.seq_len + self.horizon - 1)
        else:
            return int(self.data.shape[0] / self.seq_len)

    def __getitem__(self, idx):
        # get (12, #station, 2)
        if self.subset != 'test':
            x = self.data[idx: idx + self.seq_len]
            y = self.data[idx + self.seq_len : idx + self.seq_len + self.horizon]
            x = torch.stack((x, self.time_gen), dim=2)
            y = torch.stack((y, self.time_gen_y), dim=2)
            return x, y
        else:
            x = self.data[idx * self.seq_len : (idx + 1) * self.seq_len]
            x = torch.stack((x, self.time_gen), dim=2)
            return x, self.file_idx[idx]

if __name__ == '__main__':
    dataset = PeMSD7('data/PEMS-D7', 'train')
    print(len(dataset))
