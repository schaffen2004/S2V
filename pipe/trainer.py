from models.mae import MAE
from data.provider import provider
from tqdm import tqdm
import torch

class Trainer:
    def __init__(self):
        self.devices = 'cpu'
        self.epochs = 5
        # self.pseudo_masks = generate_pseudo_masks(args, args.batch_size)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def train_ae(self,model,dataloader):
        # mode -> train
        for epoch in tqdm(range(self.epochs)):
            for x,y in dataloader:
                inp = torch.tensor(x, dtype=torch.float32).to(self.devices)
                
                
                
        
    def train_embedd(self):
        pass
    
    def train(self):
        # Load data
        dataset,dataloader = provider()
        
        #
        self.model = MAE()
        
        self.train_ae(self.model,dataloader)
        
        