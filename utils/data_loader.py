from torch.utils.data import Dataset, DataLoader
import torch


class NumpyDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.x = train_x
        self.y = train_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float(), torch.tensor(self.y[index]).float().long()
