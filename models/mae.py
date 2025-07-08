import torch
import torch.nn as nn
from element.layer import Encoder, Interpolator,Decoder,mask_it

class MAE(nn.Module):
    def __init__(self):
        super(MAE,self).__init__()
        self.ts_size = 50
    
    def forward_mae(self, x, masks):
        """No mask tokens, using Interpolation in the latent space"""
        x_vis = mask_it(x, masks)  # (bs, vis_size, z_dim)
        x_enc = Encoder(x_vis)  # (bs, vis_size, hidden_dim)
        x_inter = Interpolator(x_enc)  # (bs, ts_size, hidden_dim)
        x_dec = Decoder(x_inter)  # (bs, ts_size, z_dim)
        return x_inter, x_dec, masks

    def forward_ae(self, x, masks):
        """mae_pseudo_mask is equivalent to the Autoencoder
            There is no interpolator in this mode"""
        x_enc = Encoder(x)
        x_dec = Decoder(x_enc)
        return x_enc, x_dec, masks

    def forward(self, x, masks, mode):

        if mode == 'train_ae':
            x_encoded, x_decoded, masks = self.forward_ae(x, masks)
        else:
            x_encoded, x_decoded, masks = self.forward_mae(x, masks)
        return x_encoded, x_decoded, masks