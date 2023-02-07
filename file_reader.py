import re

class Reader :
    def __init__(self, file_path) :
        self.file_path = file_path
        
    def read_file(self) :
        f = open(self.file_path, "r", encoding='utf-8')
        text = []
        while True :
            line = f.readline()
            text.append(line)
            if not line :
                break
        f.close()
        return text

    def make_dictionry_and_count(text) :
        stop_word_set = set([',', ' ', '-', '', '\n', '.', '!'])
        word_dict = {}  ## key-bpe encoded words / value-count : word set for calculate maximun counted pair
        
        for line in text :
            words = line.split(' ')
            print(word)
            for word in words :
                word = re.sub('[,\-\n!]', '', word)
                if word not in stop_word_set :
                    splited_word = ''
                    for character in [*word]:
                        splited_word = splited_word + character + ' '
                    splited_word = splited_word + '</w>'
                    word_dict[splited_word] = word_dict[splited_word] + 1 if (splited_word in word_dict) else 1   
        return word_dict