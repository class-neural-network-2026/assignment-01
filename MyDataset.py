import os
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from PIL import Image
import argparse
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path: str='data', split: str='train', transform=None):
        dir_data        = os.path.join(path, split)
        dir_label       = os.listdir(dir_data)
        self.list_file  = []
        self.list_label = torch.Tensor([]).to(torch.uint8)

        for i in range(len(dir_label)):
            label_data  = dir_label[i]
            path_label  = os.path.join(dir_data, label_data)
            
            if os.path.isdir(path_label) == False:
                continue
           
            file_label      = glob.glob(os.path.join(path_label, '*.png'))
            list_label      = torch.ones(len(file_label)).to(torch.uint8) * int(label_data)
            
            self.list_file  = self.list_file + file_label
            self.list_label = torch.cat((self.list_label, list_label), 0)

        self.num_data = len(self.list_file)

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
            
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        fname   = self.list_file[index]
        data    = Image.open(fname)
        data    = self.transform(data)
        label   = self.list_label[index]
        return data, label


if __name__ == '__main__':
    dataset_train = MyDataset(path='data', split='train')
    print(f'dataset (train): {len(dataset_train)}')

    dataloader_train    = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0)
    iter_data           = iter(dataloader_train)
    data, label         = next(iter_data)
    vec                 = nn.Flatten()(data)
    print(f'data (train): {data.shape}')
    print(f'label (train): {label.shape}')
    print(f'vec (train): {vec.shape}')

    plt.imshow(data[0].squeeze(), cmap='gray')
    plt.title(f'label: {label.item()}')
    plt.show()