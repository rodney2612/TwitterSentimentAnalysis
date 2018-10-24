import re

testsite_array = []
with open('data/imdb_text_all.txt',encoding='utf-8') as my_file:
    for line in my_file:
        testsite_array.append(line)

#line = file1.read()# Use this to read file content as a stream:
#words = line.split()
count=0;
appendFile = open('data/imdb_rmv_chr.txt', 'a',encoding='utf-8')

for i in range (0,len(testsite_array)):
  words=testsite_array[count].split()
  count+=1;
  for r in words:
        wr = re.sub('[^ a-zA-Z0-9]', '', r)
        appendFile.write(" "+wr.lower())

  appendFile.write("\n")



