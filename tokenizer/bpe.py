import sys
import io
from tqdm import tqdm
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf8')


class BPE_tokenizer:
    def __init__ (self, source_text, vocab_size = 12000):
        self.source_text = source_text
        self.dictionary = {} ### key: word, value: count (word is made up vocabulary tokens)
        self.vocabulary = {} ### key: bpe token(natural word), value: token id
        self.vocab_size = vocab_size
        self.__make_vocabulary(self.vocab_size)

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
                    
    def __init_vocabulary(self) :
        for word in self.dictionary :
            unit_list = word.split(' ')
            cnt = self.dictionary[word]
            for unit in unit_list :
                self.vocabulary[unit] = self.vocabulary[unit] + cnt if unit in self.vocabulary else cnt
        
    def __make_vocabulary(self, vocab_size, count=1000) :
        ### make dictionary
        ### calculate word count, storing words as sparate tokens in dictionary for bpe algorithm
        for sentence in self.source_text :
            word_list = sentence.split(' ')
            for word in word_list:
                word = word.lower()
                word = ' '.join(word)
                self.dictionary[word] = self.dictionary[word] + 1 if word in self.dictionary else 1
        
        ### make vocabulary
        # 1. initialize
        self.__init_vocabulary()
        
        # 2. BPE algorithm
        count = vocab_size - len(self.vocabulary)
        for i in tqdm(range(0, count), desc="making vocabulary", mininterval=0.1) :
            max_pair, max_count = self.__search_maxpair()
            subwords = max_pair.split(' ')
            if len(subwords) == 1 : ## bpe algorithm is converage so there is no subword for merged
                break
            frontword = subwords[0]
            backword = subwords[1]
            
            self.dictionary = self.__merge_maxpair(frontword=frontword, backword=backword)
            self.vocabulary[frontword + backword] = max_count
            self.vocabulary[frontword] = self.vocabulary[frontword] - max_count
            self.vocabulary[backword] = self.vocabulary[backword] - max_count

        print(self.dictionary)
        print(self.vocabulary)
                
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
    tokenizer = BPE_tokenizer(sorce_text)