import torch
from torch.utils.data import Dataset
import os
import json
import tiktoten 

class GPTDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name='gpt2', max_length=512, stride=1):
        self.data_dir = data_dir
        self.tokenizer = tiktoten.get_encoding(tokenizer_name)
        self.max_length = max_length
        self.stride = stride
        self.input_ids, self.target_ids = self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_dir, "the-verdict.txt"), 'r') as f:
             data = f.read()
        
        # encode data 
        tokenized_data = self.tokenizer.encode(data)

        input_tokens, target_tokens = [] , []
        for i in range(0, len(tokenized_data) - self.max_length, self.stride):
            input_chunk = tokenized_data[i: i + self.max_length]
            target_chunk = tokenized_data[i + 1: i + 1 + self.max_length]
            
            input_tokens.append(torch.tensor(input_chunk, dtype=torch.long))
            target_tokens.append(torch.tensor(target_chunk, dtype=torch.long))

 
        return input_tokens, target_tokens

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def get_dataloader(data_dir, tokenizer_name='gpt2', max_length=16, stride=8, batch_size=8, shuffle=True):
    dataset = GPTDataset(data_dir, tokenizer_name, max_length, stride)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloader