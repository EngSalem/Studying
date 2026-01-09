from self_attention_causal import selfAttentionCausal
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttentionCausal(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len, num_heads, qkv_bias=False, dropout=0.1):
        super(MultiHeadAttentionCausal, self).__init__()
        # check that multiple heads can evenly split dim_out
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads

        self.multiheads = nn.ModuleList([
            selfAttentionCausal(dim_in, self.head_dim, seq_len, qkv_bias=qkv_bias)
            for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        # get x shape
        batch_size, seq_length, dim_in = x.shape  # B x S x dim_in
        heads_output = torch.cat(
            [head(x) for head in self.multiheads], dim=-1
        )  # B x S x dim_out
        out = self.linear(heads_output)  # B x S x dim_out
        out = self.dropout(out)
        return out
    