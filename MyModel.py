import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataset import MyDataset

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        dim_input   = 1024
        dim_output  = 1
        self.weight = torch.randn((dim_input, dim_output))
        self.weight = nn.Parameter(self.weight)
        self.activate = nn.Sigmoid()
        # print(f'weight: {self.weight.shape}, {self.weight.dtype}')

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = self.activate(x)
        return x

    # =========================================================================
    # do not modify the following codes
    # =========================================================================
    def save(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        torch.save(self.state_dict(), path)

    def load(self, path='model.pth'):
        device = torch.device('cpu')
        self.to(device)
        self.load_state_dict(torch.load(path))

    def size(self):
        size = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return size
    
    def print(self):
        print(self.state_dict())
    # =========================================================================


if __name__ == '__main__':
    dataset_train = MyDataset(path='data', split='train')
    print(f'dataset (train): {len(dataset_train)}')

    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=False, num_workers=0)
    iter_data   = iter(dataloader_train)
    data, label = next(iter_data)
    vec         = nn.Flatten()(data)

    model1  = MyModel()
    model1.eval()
    model1.save()
    
    model2  = MyModel()
    model2.load()
    model2.eval()
    
    y1 = model1(vec)
    y2 = model2(vec)
    print(f'y1: {y1.shape}, {y1}')
    print(f'y2: {y2.shape}, {y2}')