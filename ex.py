import sys
print(sys.getdefaultencoding())

#file_path = 'data/de-en/train.en'
file_path = 'data/de-en/train.de'


f = open(file_path, "r", encoding='utf-8')

line = f.readline()
print(line)
print("gu")

f.close()
