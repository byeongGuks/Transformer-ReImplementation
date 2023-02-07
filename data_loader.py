import re
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Bpe_tokenizer

class TranslationDataset(Dataset):
    def __init__ (self, src_sentences, trg_sentences, src_tokenizer, trg_tokenizer, vocab_size=-1, dataset='test') :
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.IntTensor(self.src_tokenizer.generate_tokens(self.src_sentences[idx]))
        y = torch.IntTensor(self.trg_tokenizer.generate_tokens(self.trg_sentences[idx]))
        return x, y
    



