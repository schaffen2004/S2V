import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Init
    parser.add_argument('--training', type=bool, default=True, 
                        help='training mode')


    # parser.add_argument('--seed', type=int, default=42)
    
    # Paths
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='XAUUSD_M5')
    parser.add_argument('--checkpoint', type=str, default='./export/checkpoints')
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--result_path', type=str, default='./export/img')
    
    # Training hyperparameters
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

