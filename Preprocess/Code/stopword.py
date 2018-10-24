import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
import nltk
#nltk.download('stopwords')
file1 = open("english.txt")
stop_words = file1.read()
#file1 = open("char_@_rmv.txt")
testsite_array = []
with open('data/imdb_rmv_chr.txt',encoding='utf-8') as my_file:
    for line in my_file:
        testsite_array.append(line)

#line = file1.read()# Use this to read file content as a stream:
#words = line.split()
count=0;
appendFile = open('data/imdb_stopwords_text_all.txt', 'a',encoding='utf-8')

for i in range (0,len(testsite_array)):
  words=testsite_array[count].split()
  count+=1;
  for r in words:
    if not r in stop_words:
        appendFile.write(" "+r)
    else:
        print()

  appendFile.write("\n")
