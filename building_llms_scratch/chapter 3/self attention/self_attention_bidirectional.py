import torch
from torch import nn
import torch.nn.functional as F

class selfAttentionBidirectional(nn.Module):
    def __init__(self, dim_in, dim_out, qwv_bias=False):
        super(selfAttentionBidirectional, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # identify query, key, value weight matrices
        self.q = nn.Linear(dim_in, dim_out, bias=qwv_bias)
        self.k = nn.Linear(dim_in, dim_out, bias=qwv_bias)
        self.v = nn.Linear(dim_in, dim_out, bias=qwv_bias)

    def forward(self, x):
        # x_shape: (batch_size, seq_length, dim_in)
        batch_size, seq_length, dim_in = x.shape # B x S x dim_in for debugging not used in implementation

        q_w = self.q(x) # B x S x dim_out
        k_w = self.k(x) # B x S x dim_out
        v_w = self.v(x) # B x S x dim_out

        # dot product queries and keys to get raw attention scores
        attn_scores = q_w @ k_w.transpose(1, 2) # B x S x S

        # apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores/k_w.shape[-1]**0.5, dim = -1) # B x S x S

        # multiply attention weights with values to get output vectors

        out = attn_weights @ v_w # B x S x dim_out
        return out


