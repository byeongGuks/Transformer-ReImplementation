import re
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Bpe_tokenizer
import file_reader

def make_dictionry_and_count(text) :
        stop_word_set = set([',', ' ', '-', '', '\n', '.', '!'])
        word_dict = {}  ## key-bpe encoded words / value-count : word set for calculate maximun counted pair
        
        for line in text :
            words = line.split(' ')
            for word in words :
                word = re.sub('[,\-\n!]', '', word)
                if word not in stop_word_set :
                    splited_word = ''
                    for character in [*word]:
                        splited_word = splited_word + character + ' '
                    splited_word = splited_word + '</w>'
                    word_dict[splited_word] = word_dict[splited_word] + 1 if (splited_word in word_dict) else 1   
        return word_dict 

## train 과 test 경우를 다르게 처리해 주어야 함
class TranslationDataset(Dataset):
    def __init__ (self, data_path, tokenizer, vocab_size=16000, language_pair='de-en', dataset='test') :
        lang_pair = language_pair.split('-')
        file_reader_input = file_reader.Reader(data_path + 'train.' + lang_pair[0])
        sentence_input = file_reader_input.read_file()
        input_dictionary = make_dictionry_and_count(sentence_input)
        
        file_reader_output = file_reader.Reader(data_path + 'train.' + lang_pair[0])
        sentence_output = file_reader_output.read_file()
        output_dictionary = make_dictionry_and_count(sentence_output)
        
        self.src_sentences = sentence_input
        self.trg_sentences = sentence_output
        self.src_tokenizer = tokenizer(input_dictionary, vocab_size)
        self.trg_tokenizer = tokenizer(output_dictionary, vocab_size)
        
    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        x = torch.IntTensor(self.src_tokenizer.generate_tokens(self.src_sentences[idx]))
        y = torch.IntTensor(self.trg_tokenizer.generate_tokens(self.trg_sentences[idx]))
        return x, y
    
    
    
def test() :
    dataset = TranslationDataset('data/de-en/', Bpe_tokenizer, vocab_size=1000, language_pair="en-de")
    x, y = dataset.__getitem__(0)
    print(x)
    print(y)

if __name__ == "__main__":
    print("test")
    test()
    
