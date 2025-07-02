import os
from configs.config import get_args
from utils.visual import display_args
from pipe.trainer import Trainer
def main():
    # set config
    # args = get_args()
    
    # Visualize arguments info
    # display_args(args)
    
    # Init wandb
    
    # Traning
    trainer = Trainer()
    trainer.train()
       
    # Inference

if __name__=='__main__':
    main()