import file_reader
import tokenizer
import sklearn
from sklearn.model_selection import train_test_split
from tokenizer import Bpe_tokenizer
from data_loader import TranslationDataset
import argparse
from data_loader import TranslationDataset


def get_arguments() :
    parser = argparse.ArgumentParser(description='transformer argument description', prefix_chars='--')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--max_sequence_length', default=512, help="max sequence length")
    parser.add_argument('--vocab_size', default=16000, help="vocabulary size")
    parser.add_argument('--train_file_path_input', default='data/de-en/train.en', help="input text")
    parser.add_argument('--train_file_path_output', default='data/de-en/train.de', help="ouput text") ## file open 에서 codec 문제가 해결이 안됨,,,, 이유 찾기
    parser.add_argument('--test_file_path_input', default="data/de-en/test.en")
    parser.add_argument('--test_file_path_input', default="data/de-en/test.de")
    
    args = parser.parse_args()
    
    print(args.max_seq_len)    
    return args

def train(args) :
    train_dataset = TranslationDataset('data/de-en/', Bpe_tokenizer, args.vocab_size, language_pair="en-de")
    
    

def test() :
    return 0

def main(args) :
    if(args.mode == 'train') :
        train()
    if(args.mode == 'test') :
        test()

print(x_train_dictionary)

#src_tokenizer = Bpe_tokenizer(x_train_dictionary, VOCAB_SIZE)
#trg_tokenizer = Bpe_tokenizer(y_train_dictionary, VOCAB_SIZE)


#dataset = TranslationDataset(src_sentences=sentence_en, trg_sentences=sentence_de, src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, vocab_size=VOCAB_SIZE)
#print(dataset.__getitem__(10))
     

#bpe_tokenizer = tokenizer.Bpe_tokenizer(word_dict, vocab_size = 100)

#test 
#print(word_dict)
#print(bpe_tokenizer.generate_tokens('low lower newest widest'))