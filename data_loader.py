import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, x, y, dataset='test'):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x.iloc[idx])
        y = torch.FloatTensor(self.y.iloc[idx])
        return x, y
    

dataset = TranslationDataset(x= [1,1], y=[1,1])


