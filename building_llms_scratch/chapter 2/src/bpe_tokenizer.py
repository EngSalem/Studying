import re 
from typing import List
from collections import Counter, defaultdict

def corpus_reader(file_path: str) -> List[str]:
    """
    file_path: corpus path
    return: list of instances in string format
    """
    return open(file_path).readlines()

def text_reader(file_path: str) -> str:
    """
    file_path: full text path
    return: all text as a a sngle string
    """
    return open(file_path).read()

def get_stats(vocab):
    ##
    # get the most frequent pairs  
    ##
    # get adjacent pairs
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for ix in range(len(word)-1):
            pairs[(word[ix],word[ix+1])] += freq

    return Counter(pairs).most_common(1)        


   
def merge(vocab, most_freq_pair):
    merged_vocab = {}
    replacement = most_freq_pair[0]+most_freq_pair[1]
    for word, freq in vocab.items():
        new_word = []
        ix = 0
        while ix<len(word)-1:
            if (word[ix], word[ix+1]) == most_freq_pair:
                new_word.append(replacement)
                ix+=2
            else:
                new_word.append(word[ix])
                ix+=1  
        if ix == len(word) - 1:
           new_word.append(word[ix])
        

        merged_vocab[tuple(new_word)]= freq        
   
    return merged_vocab
     
              
    


class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size ## stoping merge
        self.token_map = dict()
        self.id2token = dict()
    
    def build_tokenizer(self, corpus_path: str) -> None:
        text = text_reader(corpus_path)

        ## convert into a list of characters 
        words = text.split() ## split om space

        ## init vocabulary
        vocab = {list(word)+["</w>"]: words.count(word) for word in words}

        



        while len(vocab) < self.vocab_size:
            ## merge till we get the vocabulary size
            # step 1 get stats to get most frequent pairs
            most_freq_pair =  get_stats(vocab)

            # step 2 get 
            vocab =  merge(vocab, most_freq_pair)




    



        corpus = text_reader(corpus_path)
        splitted_tokens = _tokenize_text(corpus)

        tokenmap,id2token, vocabulary_len = _build_tokenmap(splitted_tokens)
        self.token_map = tokenmap
        self.id2token = id2token
        self.vocab_size = vocabulary_len

    def encode(self, text_instance : str) -> List[int]:
        if len(self.token_map) == 0:
            raise ValueError("Tokenizer is not built and empty build it first by calling SimpleTokenizer.build_tokenizer")

        ## split text then encode
        instance_tokens =  re.split(r'([,.:;?_!"()\']|--|\s)', text_instance)
        return [self.token_map[_token] if _token in self.token_map else "<|unk|>" for _token in instance_tokens ] ## return

    def decode(self, text_ids: List[int]) -> str:
        return " ".join([self.id2token[_id] for _id in text_ids])


