import sys
import io
from tqdm.auto import tqdm
import json
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf8')


class BPE_tokenizer:
    def __init__ (self, source_text, vocab_size = 12000, vocab_L2T_filepath=None, vocab_T2L_filepath=None):
        self.source_text = source_text
        self.dictionary = {} ### key: word, value: count (word is made up vocabulary tokens)
        self.vocabulary_L2T = {} ### key: bpe token(natural word), value: token id
        self.vocabulary_T2L = [] ### index: token_id, value: subword
        self.vocab_count = {} ### key : subword, value: count
        self.vocab_size = vocab_size
        
        if (vocab_T2L_filepath!=None) and (vocab_L2T_filepath!=None) :
            try :
                with open(vocab_L2T_filepath, 'r') as file :
                    data = json.load(file)
                    self.vocabulary_L2T = data
                with open(vocab_T2L_filepath, 'r') as file :
                    data = json.load(file)
                    self.vocabulary_T2L = data
            except : ## for unvalid file path
                self.__make_vocabulary(vocab_size=self.vocab_size)
        else :
            self.__make_vocabulary(vocab_size=self.vocab_size)
            
        self.WHITE_SPACE = '</w>'
        self.UNKNOWN_TOKEN = 2
    
    def __len__(self) :
        return len(self.vocabulary_L2T)

    def __search_maxpair (self) :
        # 1. pair count
        pair_count = {}
        for word in self.dictionary :
            unit_list = word.split(' ')
            cnt = self.dictionary[word]
            
            for i in range(len(unit_list) - 1) :
                new_word = unit_list[i] + ' ' + unit_list[i+1] ## insert space for distinguish frontword and backword
                pair_count[new_word] = pair_count[new_word] + cnt if (new_word in pair_count) else cnt
        
        # 2. find max counted pair
        max_count = 0
        max_pair = ''
        for pair in pair_count:
            if pair_count[pair] > max_count :
                max_count = pair_count[pair]
                max_pair = pair
        return (max_pair, max_count)
    
    def __merge_maxpair (self, frontword, backword) :
        new_word_dict = {}        
        for word in self.dictionary :
            unit_list = word.split(' ')
            new_word = ''
            i=0
            while i < len(unit_list) :
                if i == len(unit_list)-1 or unit_list[i] != frontword or unit_list[i+1] != backword : 
                    new_word = new_word + unit_list[i] + ' '
                else :
                    new_word = new_word + unit_list[i] + unit_list[i+1] + ' '
                    i += 1 ## skip next unit
                i += 1
            new_word = new_word[:-1]
            new_word_dict[new_word] = self.dictionary[word] ## change word with new key
        return new_word_dict
                    
    def __init_vocab_count(self) :
        for word in self.dictionary :
            unit_list = word.split(' ')
            cnt = self.dictionary[word]
            for unit in unit_list :
                self.vocab_count[unit] = self.vocab_count[unit] + cnt if unit in self.vocab_count else cnt
        
    def __make_vocabulary(self, vocab_size) :
        ### make dictionary
        ### calculate word count, storing words as sparate tokens in dictionary for bpe algorithm
        for sentence in self.source_text :
            word_list = sentence.split(' ')
            for word in word_list:
                word = word.lower()
                word = ' '.join(word) + ' ' + '</w>' ### self.WHITE_SPACE
                self.dictionary[word] = self.dictionary[word] + 1 if word in self.dictionary else 1
        
        ### make vocabulary
        # 1. initialize
        self.__init_vocab_count()
        
        # 2. BPE algorithm
        count = vocab_size - len(self.vocab_count)
        for i in tqdm(range(count), desc="making vocabulary", position=0, leave=False) :
            max_pair, max_count = self.__search_maxpair()
            subwords = max_pair.split(' ')
            if len(subwords) == 1 : ## bpe algorithm is converage so there is no subword for merged
                break
            frontword = subwords[0]
            backword = subwords[1]
            
            self.dictionary = self.__merge_maxpair(frontword=frontword, backword=backword)
            self.vocab_count[frontword + backword] = max_count
            self.vocab_count[frontword] = self.vocab_count[frontword] - max_count
            self.vocab_count[backword] = self.vocab_count[backword] - max_count
        
        # 3. bulid vocabulary
        self.vocabulary_L2T['<pad>'] = 0
        self.vocabulary_L2T['<bos>'] = 1
        self.vocabulary_L2T['<unk>'] = 2
        
        self.vocabulary_T2L = ['<pad>', '<bos>', '<unk>']
        id = 3
        for subword in self.vocab_count :
            self.vocabulary_L2T[subword] = id
            self.vocabulary_T2L.append(subword)
            id += 1
             
    def tokenize(self, word):
        word = word
        token_sequence = []
        
        pre_word = ""
        curr_token = -1
        ## find logest subword that enrolled vocabulary started i position
        i = 0
        while i < len(word) :
            for j in range(len(word) - i) :
                end_position = len(word) - j
                sub_word = word[i:end_position]
                if sub_word in self.vocabulary_L2T :
                    token_sequence.append(self.vocabulary_L2T[sub_word])
                    i = end_position - 1
                    break
                if j == (len(word) - i) - 1 : ## 마지막 까지 match 안됐다면 unknown token
                    token_sequence.append(self.UNKNOWN_TOKEN)
                    
            i+=1
        token_sequence.append(self.vocabulary_L2T[self.WHITE_SPACE])     
        return token_sequence
    
    def encode(self, sentences):
        sentences = sentences.lower()
        token_sequence = []
        words = sentences.split(' ')
        for word in words :
            token_sequence += self.tokenize(word)
        return token_sequence
    
    def decode(self, token_sequence):
        sentence = ""
        for token in token_sequence :
            sentence += self.vocabulary_T2L[token]
        return sentence

                
if __name__ == "__main__":
    print("test")
    sorce_text = [
        "Room-temperature superconductivity has long been the holiest of holy grails in condensed-matter physics.", 
        "Within the past decade, the appearance of new materials that superconduct at relatively balmy temperatures,",
        "but only under extreme pressures, has brought a slight yet significant alteration in the quest.", 
        "To be truly grail-like, a newly synthesized superconductor cannot merely carry electrical current without resistance at room temperature."
        "It must also do it at ambient pressure for it to have practical applications beyond the laboratory – such as levitating trains,",
        "efficient power lines or cheaper MRI machines. or"
    ]
    tokenizer = BPE_tokenizer(sorce_text, vocab_size=60)
    print(tokenizer.dictionary)
    print(tokenizer.vocab_count)
    print(tokenizer.vocabulary_L2T)
    print(tokenizer.vocabulary_T2L)
    
    example_sentence = "efficient power lines or cheaper MRI machines. or"
    encode_sentence = tokenizer.encode(example_sentence)
    print(encode_sentence)
    print(tokenizer.decode(encode_sentence))
    print(tokenizer.__len__())
    



