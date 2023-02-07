import file_reader
import tokenizer
import sklearn
from sklearn.model_selection import train_test_split
from tokenizer import Bpe_tokenizer
from data_loader import TranslationDataset


VOCAB_SIZE = 1000
file_path_en = 'data/de-en/train.en'
file_path_de = 'data/de-en/train.en' #'data/de-en/train.de' ## file open 에서 codec 문제가 해결이 안됨,,,, 이유 찾기
##file_path = 'data/de-en/bpe_ex.en'

file_reader_en = file_reader.Reader(file_path_en)
sentence_en = file_reader_en.read_file()


file_reader_de = file_reader.Reader(file_path_de)
sentence_de = file_reader_de.read_file()

x_train, x_valid, y_train, y_valid = train_test_split(sentence_en, sentence_de, test_size=0.2, shuffle=True)

x_train_dictionary = file_reader_en.make_dictionry_and_count(sentence_en)
y_train_dictionary = file_reader_de.make_dictionry_and_count(sentence_de)

print(x_train_dictionary)

#src_tokenizer = Bpe_tokenizer(x_train_dictionary, VOCAB_SIZE)
#trg_tokenizer = Bpe_tokenizer(y_train_dictionary, VOCAB_SIZE)


#dataset = TranslationDataset(src_sentences=sentence_en, trg_sentences=sentence_de, src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, vocab_size=VOCAB_SIZE)
#print(dataset.__getitem__(10))
     

#bpe_tokenizer = tokenizer.Bpe_tokenizer(word_dict, vocab_size = 100)

#test 
#print(word_dict)
#print(bpe_tokenizer.generate_tokens('low lower newest widest'))