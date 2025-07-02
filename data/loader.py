from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class XAUUSD(Dataset):
    def __init__(self):
        super(XAUUSD).__init__()
        self.data = self.load_data()
        self.ts_size = 24
        self.pred_len = 1
    
    def load_data(self):
        path = "/home/schaffen/Workspace/Project/TS/S2V/data/XAUUSD_M15.csv"
        # Load all columns as string to handle datetime
        data = np.loadtxt(path, delimiter=',', dtype=str, skiprows=1)
        data = data[:,1:].astype(float)
        return data

    def __getitem__(self, index):
        self.begin = index
        self.end = self.begin + self.ts_size
        self.x = self.data[self.begin:self.end]
        self.y = self.data[self.end:self.end+self.pred_len]
        return self.x, self.y
    
    def __len__(self):
        return len(self.data) - self.ts_size
    
    