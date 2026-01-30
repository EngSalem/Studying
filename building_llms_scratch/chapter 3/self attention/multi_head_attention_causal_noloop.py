import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttentionCausal(nn.Module):
    def __init__(self, dim_in, dim_out, seq_len, num_heads, qkv_bias=False, dropout=0.1):
        super(MultiHeadAttentionCausal, self).__init__()
        # dimensions and parameters
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.seq_len = seq_len
        self.qkv_bias = qkv_bias

        # linear layers for query, key, value
        self.q = nn.Linear(dim_in, dim_out, bias = qkv_bias)
        self.k = nn.Linear(dim_in, dim_out, bias = qkv_bias)
        self.v = nn.Linear(dim_in, dim_out, bias = qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_out, dim_out) ## aggregation linear layer

        # causal mask to prevent attention to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)) # (1, 1, seq_len, seq_len)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension

        # compute queries, keys, values
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # reshape for multi-head attention
        # multiply by num_heads and head_dim
        q = q.view(B,T, self.num_heads, self.head_dim)
        k = k.view(B,T, self.num_heads, self.head_dim)
        v = v.view(B,T, self.num_heads, self.head_dim)
        attention_scores = (q.transpose(1,2) @ k.transpose(1,2).transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attention_scores = attention_scores.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))  # apply causal mask

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, T, T)
        attention_weights = self.attn_drop(attention_weights)

        out = attention_weights @ v.transpose(1,2)  # (B, num_heads, T, head_dim)
        context_vector = out.transpose(1,2).contiguous().view(B, T, self.dim_out)

        # apply projection [optional]
        context_vector = self.proj(context_vector)
        return context_vector
