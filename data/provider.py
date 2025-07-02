from data.loader import XAUUSD
from torch.utils.data import DataLoader

data_dict = {
    "XAUUSD": XAUUSD,
}

def provider():
    dataset = data_dict["XAUUSD"]()
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    return dataset,dataloader