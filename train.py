import argparse
import json
from tqdm import tqdm
from tokenizer.bpe import BPE_tokenizer
from translation_dataset import TranslationDataset
from torch.utils.data import DataLoader



def get_arguments() :
    parser = argparse.ArgumentParser(description='transformer argument description', prefix_chars='--')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--max_sequence_length', default=512, help="max sequence length")
    parser.add_argument('--vocab_size', default=100, help="vocabulary size")
    parser.add_argument('--data_path', default='./data/translation/en-de.json')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epoch', default=100)
    
    args = parser.parse_args()
    return args

def train(args) :
    print(args.data_path) 
    with open(args.data_path, 'r', encoding='utf-8') as file:
        source_data = json.load(file)
        
    en_sentences = []
    de_sentences = []
    for translation_pair in source_data :
        en_sentences.append(translation_pair['en'])
        de_sentences.append(translation_pair['de'])
    
    src_tokenizer = BPE_tokenizer(en_sentences)
    trg_tokenizer = BPE_tokenizer(de_sentences)
    
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
    
    
    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader):
            print(data)
            break

def main(args) :
    if(args.mode == 'train') :
        train()
    if(args.mode == 'test') :
        test()


if __name__ == "__main__":
    print("test")
    train(get_arguments())