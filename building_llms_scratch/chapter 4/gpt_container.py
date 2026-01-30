import torch 
from torch import nn
from building_llms_scratch.chapter_3.self_attention.multi_head_attention_causal_noloop import MultiHeadAttentionCausal

class DummyLayerNorm(nn.Module):
    def __init__(self, norm_dim, eps=1e-5):
        super(DummyLayerNorm, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x): ## does nothing for now
        return x    

class GPTContainer(nn.Module):
    def __init__(self,config: dict):
        super(GPTContainer, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.position_embedding = nn.Embedding(config['max_seq_len'], config['embed_dim'])
        self.final_layer_norm = DummyLayerNorm(config['embed_dim']) # Dummy LayerNorm as placeholder
        self.transoformer_blocks = nn.Sequential([MultiHeadAttentionCausal(
            dim_in=config['embed_dim'],
            dim_out=config['embed_dim'],
            seq_len=config['max_seq_len'],
            num_heads=config['num_heads'],
            qkv_bias=config.get('qkv_bias', False),
            dropout=config.get('dropout', 0.1)
        ) for _ in range(config['num_layers'])])

        self.projection_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=False) 
        ## This is because gpt outputs embeddings mainly , we want to transform them back to vocab size for prediction

    def forward(self, x):
        B, T = x.shape # batch_size, seq_len
        token_embeddings = self.token_embedding(x) # (B, T, embed_dim)
        pos_embeggings = self.position_embedding(torch.arange(T, device=x.device)) # (T, embed_dim)
        x = token_embeddings + pos_embeggings.unsqueeze(0) # (B, T, embed_dim)
        x = self.transoformer_blocks(x) # (B, T, embed_dim)
        x = self.final_layer_norm(x) # (B, T, embed_dim)
        logits = self.projection_head(x) # (B, T, vocab_size)
        return logits
