import os
import h5py
import numpy
import torch

from torch.utils.data import Dataset, DataLoader

class WeightDataset(Dataset):
    def __init__(self, file_list, margin, dim, data_dir, save_dir, subset, num_classes):
        self.file_list = file_list
        self.DATA_DIR = data_dir
        self.SAVE_DIR = save_dir
        self.margin = margin
        self.dim = dim
        self.subset = subset
        self.num_classes = num_classes

    def __getitem__(self, index):
        idx = int(index/2048)
        idx2 = index%2048

        label_path = os.path.join(self.DATA_DIR, self.file_list[idx])
        label_data = h5py.File(label_path, 'r')['label'][idx2][0]

        h5name = self.file_list[idx]+ 'margin' + str(self.margin) + 'dim' + str(self.dim) + '.h5'
        weight_name = os.path.join(self.SAVE_DIR, h5name)
        weight = h5py.File(weight_name, 'r')['data'][idx2]

        all_idx = numpy.arange(2048)
        numpy.random.shuffle(all_idx)
        final_idx = all_idx[:self.subset]
        weight = weight[final_idx,:]

        return {'data':torch.Tensor(weight), 'label':label_data}

    def __len__(self):

        cnt = 0
        for file in self.file_list:
            path = os.path.join(self.DATA_DIR, file)
            d = h5py.File(path, 'r')
            cnt += d['data'].shape[0]
            d.close()
        return cnt

def get_dataloader(file_list, margin, dim, phase, batch_size, DATA_DIR, SAVE_DIR, subset, num_classes, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = WeightDataset(file_list, margin, dim, DATA_DIR, SAVE_DIR, subset, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=4)
    return dataloader

# TODO
class WeightPNDataset(Dataset):
    def __init__(self, file_list, margin, dim, data_dir, save_dir, subset, num_classes):
        self.file_list = file_list
        self.DATA_DIR = data_dir
        self.SAVE_DIR = save_dir
        self.margin = margin
        self.dim = dim
        self.subset = subset
        self.num_classes = num_classes

    def __getitem__(self, index):
        idx = int(index/2048)
        idx2 = index%2048

        label_path = os.path.join(self.DATA_DIR, self.file_list[idx])
        label_data = h5py.File(label_path, 'r')['label'][idx2][0]

        h5name = self.file_list[idx]+ 'margin' + str(self.margin) + 'dim' + str(self.dim) + '.h5'
        weight_name = os.path.join(self.SAVE_DIR, h5name)
        weight = h5py.File(weight_name, 'r')['data'][idx2]

        all_idx = numpy.arange(2048)
        numpy.random.shuffle(all_idx)
        final_idx = all_idx[:self.subset]
        weight = weight[final_idx,:]

        return {'data':torch.Tensor(weight), 'label':label_data}

    def __len__(self):

        cnt = 0
        for file in self.file_list:
            path = os.path.join(self.DATA_DIR, file)
            d = h5py.File(path, 'r')
            cnt += d['data'].shape[0]
            d.close()
        return cnt