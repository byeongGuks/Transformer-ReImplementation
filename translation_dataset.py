import torch
from torch.utils.data import Dataset, DataLoader
import json

class TranslationDataset(Dataset) :
    def __init__ (self, data_path, src_tokenizer, trg_tokenizer, src_lang='en', trg_lang='de', model_dimension=512) :
        with open(data_path, 'r', encoding='utf-8') as file:
            self.sentences = json.load(file)
        
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.model_dimension = model_dimension
    
    def __len__(self) :
        return len(self.sentences)
    
    def __getitem__(self, idx) :
        x = torch.LongTensor(self.src_tokenizer.encode(self.sentences[idx][self.src_lang]))
        y = torch.LongTensor([1] + self.trg_tokenizer.encode(self.sentences[idx][self.trg_lang])) ## add bos token
        ## 1 == self.trg_tokenizer.vocabulary_L2T['<bos>']
        padded_x = torch.nn.functional.pad(input = x, pad = (0, self.model_dimension-x.size(dim=0)), mode='constant', value=0) ## 512
        padded_y = torch.nn.functional.pad(input = y, pad = (0, self.model_dimension-y.size(dim=0)), mode='constant', value=0) ## 512 bos token 자리 남겨둠 
           
        return {"input": padded_x, "output": padded_y}
    
    def collate_fn(self, samples):
        inputs = [sample['input'] for sample in samples]
        outputs = [sample['output'] for sample in samples]
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        padded_outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        return {'input' : padded_inputs.contiguous(),
                'output' : padded_outputs.contiguous()}              
      
if __name__ == "__main__":
    print("test")
    source_text = [
        "Room-temperature superconductivity has long been the holiest of holy grails in condensed-matter physics.", 
        "Within the past decade, the appearance of new materials that superconduct at relatively balmy temperatures,",
        "but only under extreme pressures, has brought a slight yet significant alteration in the quest.", 
        "To be truly grail-like, a newly synthesized superconductor cannot merely carry electrical current without resistance at room temperature."
        "It must also do it at ambient pressure for it to have practical applications beyond the laboratory – such as levitating trains,",
        "efficient power lines or cheaper MRI machines. or"
    ]
    
    import sys
    from tokenizer.bpe import BPE_tokenizer

    src_tokenizer = BPE_tokenizer(source_text)
    trg_tokenizer = BPE_tokenizer(source_text)
    translation_dataset = TranslationDataset(data_path='./data/translation/en-de.json', src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, model_dimension=64)
    print(translation_dataset.__getitem__(0))
    print(len((translation_dataset.__getitem__(0)['input'])))
    print(len((translation_dataset.__getitem__(0)['output'])))
    
    ### full text test code 
    """
    from bpe import BPE_tokenizer
    
    import json
    file_path = './data/translation/en-de.json' 
    with open(file_path, 'r', encoding='utf-8') as file:
        source_data = json.load(file)
        
    en_sentences = []
    de_sentences = []
    for translation_pair in source_data :
        en_sentences.append(translation_pair['en'])
        de_sentences.append(translation_pair['de'])
    
    src_tokenizer = BPE_tokenizer(en_sentences)
    trg_tokenizer = BPE_tokenizer(de_sentences)
    
    translation_dataset = TranslationDataset(data_path='./data/translation/en-de.json', src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer)
    """
