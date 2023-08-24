import argparse
import json
from tqdm import tqdm
from tokenizer.bpe import BPE_tokenizer
from translation_dataset import TranslationDataset
from torch.utils.data import DataLoader
from translation_model.transformer import Transformer
import torch.nn as nn
import torch.optim as optim


def get_arguments() :
    parser = argparse.ArgumentParser(description='transformer argument description', prefix_chars='--')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--max_sequence_length', default=512, help="max sequence length")
    parser.add_argument('--vocab_size', default=16000, help="vocabulary size")
    parser.add_argument('--data_path', default='./data/translation/en-de.json')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--learning_rate', default=1e-03, type=float) 
    
    args = parser.parse_args()
    return args

def train(args) :
    with open(args.data_path, 'r', encoding='utf-8') as file:
        source_data = json.load(file)
        
    en_sentences = []
    de_sentences = []
    for translation_pair in source_data :
        en_sentences.append(translation_pair['en'])
        de_sentences.append(translation_pair['de'])
    
    src_tokenizer = BPE_tokenizer(en_sentences, 
                                  vocab_L2T_filepath='C:/Users/DMIS/project/transformer/data/de-en/vocabulary/src_vocab_L2T',
                                  vocab_T2L_filepath='C:/Users/DMIS/project/transformer/data/de-en/vocabulary/src_vocab_T2L')
    trg_tokenizer = BPE_tokenizer(de_sentences,
                                  vocab_L2T_filepath='C:/Users/DMIS/project/transformer/data/de-en/vocabulary/trg_vocab_L2T',
                                  vocab_T2L_filepath='C:/Users/DMIS/project/transformer/data/de-en/vocabulary/trg_vocab_T2L')
    
    #with open('./data/de-en/vocabulary/src_vocab_L2T', 'w') as file:
    #    json.dump(src_tokenizer.vocabulary_L2T, file)
    #with open('./data/de-en/vocabulary/trg_vocab_L2T', 'w') as file:
    #    json.dump(trg_tokenizer.vocabulary_L2T, file)
    #with open('./data/de-en/vocabulary/src_vocab_T2L', 'w') as file:
    #    json.dump(src_tokenizer.vocabulary_T2L, file)
    #with open('./data/de-en/vocabulary/trg_vocab_T2L', 'w') as file:
    #    json.dump(trg_tokenizer.vocabulary_T2L, file)
    
    translation_dataset = TranslationDataset(data_path=args.data_path,
                                 src_tokenizer=src_tokenizer,
                                 trg_tokenizer=trg_tokenizer,
                                 model_dimension=512)
    
    dataloader = DataLoader(dataset=translation_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=translation_dataset.collate_fn)
    
    model = Transformer(d_model=512, 
                        n_head=8, 
                        n_encoder=6, 
                        n_decoder=6,
                        src_vocab_size=src_tokenizer.__len__(),
                        trg_vocab_size=trg_tokenizer.__len__())
    
    criterion = nn.NLLLoss(ignore_index=0) ##pad token
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,betas=(0.9, 0.98), eps=1e-09)
    
    model.train()
    for epoch in range(args.epoch):
        train_loss = 0
        for i, data in enumerate(dataloader):
            src = data['input']
            trg = data['output']
            
            model.zero_grad()
            outputs = model(src, trg)
            
            print(outputs.view(-1, trg_tokenizer.__len__()).size())
            print(trg.view(-1).size())
            
            loss = criterion(outputs.view(-1, trg_tokenizer.__len__()), trg.view(-1))
            #print(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            
            print(loss.size())
            print(loss)
            print(loss.item)
            print(train_loss)
            break
        break

def main(args) :
    if(args.mode == 'train') :
        train()
    if(args.mode == 'test') :
        test()


if __name__ == "__main__":
    print("test")
    train(get_arguments())