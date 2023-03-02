import file_reader
import tokenizer
import torch
from sklearn.model_selection import train_test_split
from tokenizer import Bpe_tokenizer
from data_loader import TranslationDataset
import argparse
from data_loader import TranslationDataset
from torch.utils.data import DataLoader
from model import TransFormerModel


def get_arguments() :
    parser = argparse.ArgumentParser(description='transformer argument description', prefix_chars='--')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--max_sequence_length', default=512, help="max sequence length")
    parser.add_argument('--vocab_size', default=16000, help="vocabulary size")
    parser.add_argument('--train_file_path_input', default='data/de-en/train.en', help="input text")
    parser.add_argument('--train_file_path_output', default='data/de-en/train.de', help="ouput text") ## file open 에서 codec 문제가 해결이 안됨,,,, 이유 찾기
    parser.add_argument('--test_file_path_input', default="data/de-en/test.en")
    parser.add_argument('--test_file_path_input', default="data/de-en/test.de")
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epoch', default=100)
    
    args = parser.parse_args()
    
    print(args.max_seq_len)    
    return args

def train(args) :
    train_dataset = TranslationDataset('data/de-en/', Bpe_tokenizer, args.vocab_size, language_pair="en-de")
    
    data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, suffle=True, collate_fn=train_dataset.collate_fn)
    
    model = TransFormerModel(model_dimension = 512, num_head = 8, num_encoder = 6, num_decoder = 6, vocab_size=32000)
    
    ## Loss function
    criterian = torch.nn.NLLLoss(ignore_index=0) ## padding ignore
    
    ## Optimizer
    lr = (512 ** -0.5) * min()
    optimizer = torch.optim.Adam(betas=(0.9, 0.98), eps=1e-9)
    
    ## training
    for epoch in range(args.epoch):
        train_loss = 0
        train_total = 0
        model.train()
        for i, data in enumerate(data_loader):
            x, y = data
    
    
    
    

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