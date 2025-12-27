import re 
from typing import List

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


class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = None
        self.token_map = dict()
        self.id2token = dict()
    
    def build_tokenizer(self, corpus_path: str) -> None:
        def _tokenize_text(full_text):
            splitted_corpus = re.split(r'([,.:;?_!"()\']|--|\s)', full_text) ## pattern from the book ## regex101 check https://regex101.com/
            return [token.strip() for token in splitted_corpus if token.strip()]
        
        def _build_tokenmap(tokens: List[str]):
             vocabulary = sorted(list(set(tokens)))
             vocabulary.extend(["<|endoftext|>", "<|unk|>"])
             vocab2id = {_token:_id for _id,_token in enumerate(vocabulary)}
             id2vocab = {_id:_token for _token,_id in vocab2id.items()}
             return vocab2id,id2vocab, len(vocabulary)

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


