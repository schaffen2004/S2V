import os
from configs.config import get_args
from utils.visual import display_args

def main():
    # set config
    args = get_args()
    
    # Visualize arguments info
    display_args(args)
    
    # Init wandb
    
    # Traning
    
    # Inference

if __name__=='__main__':
    main()