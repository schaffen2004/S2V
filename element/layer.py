import torch.nn as nn

from tqdm import tqdm
from einops import rearrange


def mask_it(x, masks):
    # x(bs, ts_size, z_dim)
    b, l, f = x.shape
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, vis_size, z_dim)
    return x_visible


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)
        return x_dec


class Interpolator(nn.Module):
    def __init__(self, args):
        super(Interpolator, self).__init__()
        self.sequence_inter = nn.Linear(in_features=(args.ts_size - args.total_mask_size),
                                        out_features=args.ts_size)
        self.feature_inter = nn.Linear(in_features=args.hidden_dim,
                                       out_features=args.hidden_dim)

    def forward(self, x):
        # x(bs, vis_size, hidden_dim)
        x = rearrange(x, 'b l f -> b f l')  # x(bs, hidden_dim, vis_size)
        x = self.sequence_inter(x)  # x(bs, hidden_dim, ts_size)
        x = rearrange(x, 'b f l -> b l f')  # x(bs, ts_size, hidden_dim)
        x = self.feature_inter(x)  # x(bs, ts_size, hidden_dim)
        return x
