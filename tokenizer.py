import re
from tqdm import tqdm
from tqdm import trange


'''
init : word-count dictionary 
'''
class Bpe_tokenizer:
    ## vocab_size -1 mean using default vocabulary size
    def __init__(self, src_word_dict, vocab_size = -1):
        self.word_dict = src_word_dict
        self.vocabulary = {}
        self.vocab_size = vocab_size
        self.__make_vocabulary(vocab_size)
        self.UNKNOWN_TOKEN = 2
        self.WHITE_SPACE = '</w>'
        self.tokens = self.__make_tokens()
        #print(self.tokens)

    
    ## return (maximum counted pair, count)
    ## ex - (h st, 10)    
    def __search_max_pair(self, dict) :
        pair_count = {}
        
        ## pair count
        for word in dict:
            unit_list = word.split(' ')
            count_of_word = dict[word] 
            for i in range(len(unit_list) - 1):
                new_word = unit_list[i] + ' ' + unit_list[i+1] ## insert space for distinguish frontword and backword
                pair_count[new_word] = pair_count[new_word] + count_of_word if (new_word in pair_count) else count_of_word
        ## search maximum pair
        max_count = 0
        max_pair = ''
        for pair in pair_count:
            if pair_count[pair] > max_count :
                max_count = pair_count[pair]
                max_pair = pair
        return (max_pair, max_count)
    
    def __merge_word_dict(self, dict, frontword, backword) :
        new_word_dict = {}
        for word in dict :
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
            new_word_dict[new_word] = dict[word] ## change word with new key
        return new_word_dict
                
    
    def __make_vocabulary(self, vocab_size, count=1000) :
        for word in self.word_dict :
            unit_list = word.split(' ')
            for unit in unit_list:
                self.vocabulary[unit] = self.vocabulary[unit] + self.word_dict[word] if (unit in self.vocabulary) else self.word_dict[word]
        
        if vocab_size != -1 :
            count = vocab_size - len(self.vocabulary) 
             
        for i in tqdm(range(count), desc="making vocabulary", mininterval=0.1) :
            max_pair, max_count = self.__search_max_pair(self.word_dict)
            subwords = max_pair.split(' ')
            if len(subwords) == 1 : ## bpe algorithm is converage so there is no subword for merged
                break
            frontword = subwords[0]
            backword = subwords[1]
            
            self.word_dict = self.__merge_word_dict(self.word_dict, frontword=frontword, backword=backword)
            self.vocabulary[frontword + backword] = max_count
            self.vocabulary[frontword] = self.vocabulary[frontword] - max_count
            self.vocabulary[backword] = self.vocabulary[backword] - max_count
    
    def __make_tokens(self) :
        tokens = {}
        
        tokens['<pad>'] = 0
        tokens['bos'] = 1
        tokens['<unk>'] = 2
        id = 2
        for word in self.vocabulary :
            tokens[word] = id
            id += 1
        return tokens
            
    ## word to token sequence 
    def tokenize(self, word):
        tokens = []
        pre_word = ""
        curr_token = -1
        ## find logest subword that enrolled vocabulary started i position
        i = 0
        while i < len(word) :
            for j in range(len(word) - i) :
                end_position = len(word) - j
                sub_word = word[i:end_position]
                if sub_word in self.tokens :
                    tokens.append(self.tokens[sub_word])
                    i = end_position - 1
                    break
                if j == (len(word) - i) - 1 : ## 마지막 까지 match 안됐다면 unknown token
                    tokens.append(self.UNKNOWN_TOKEN)
            i+=1        
        return tokens
    
    ## sentence to words with cleaning
    def generate_tokens(self, sentence) :
        tokens = []
        words = sentence.split(' ')
        for word in words :
            tokens = tokens + self.tokenize(word + self.WHITE_SPACE) ## </w> 단어 끝에 붙여서 encoding
        return tokens


    #def untokenize():