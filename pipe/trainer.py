from models.mae import MAE
from data.provider import provider

class Trainer:
    def __init__(self):
        self.devices = 'cpu'
        self.model = MAE()
    
    def train_ar(self):
        # mode -> train
        pass
        
    def train_embedd(self):
        pass
    
    def train(self):
        dataset,dataloader = provider()
        
        for x,y in dataloader:
            print(x.shape,y.shape)
            break