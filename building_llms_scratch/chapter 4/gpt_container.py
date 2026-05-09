import torch 
from torch import nn
from building_llms_scratch.chapter_3.self_attention.multi_head_attention_causal_noloop import MultiHeadAttentionCausal

class DummyLayerNorm(nn.Module):
    def __init__(self, norm_dim, eps=1e-5):
        super(DummyLayerNorm, self).__init__()
        self.norm_dim = norm_dim

    def forward(self, x): ## does nothing for now
        return x    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.emb_dim = emb_dim
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        ##
        # apply normalization with biased variance and mean
        # #    
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift ## helps network learn optimal scale and shift for normalized output

class DummyTransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super(DummyTransformerBlock, self).__init__()
        self.config = config

    def forward(self, x):
        return x  ## does nothing for now    

class GPTContainer(nn.Module):
    def __init__(self,config: dict):
        super(GPTContainer, self).__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.position_embedding = nn.Embedding(config['max_seq_len'], config['embed_dim'])
        self.final_layer_norm = LayerNorm(config['embed_dim']) # Use the implemented LayerNorm
        self.transoformer_blocks = nn.Sequential([DummyTransformerBlock(config
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
