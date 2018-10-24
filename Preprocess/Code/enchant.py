print("start")
import csv
import itertools
from numpy import ndarray
from scipy.spatial import distance

import numpy as np
import math
import pandas as pd
import networkx as nx

#import enchant
#from ruleslangdetection import RuleSlangDetection
reader = csv.reader(open("data/slangdict.csv", "rt"))#, delimiter="   ")
x = list(reader)
r = np.array(x)
print(r[500][0])
print(r[500][1])
testsite_array = []

with open('char_@_rmv.txt',encoding='utf-8') as my_file:
    for line in my_file:
        testsite_array.append(line)

#line = file1.read()# Use this to read file content as a stream:
#words = line.split()
count=0;
appendFile = open('Keep_@_spell_check.txt', 'a',encoding='utf-8')

for i in range (0,len(testsite_array)):
  words=testsite_array[count].split()
  count+=1;
  flag=0
  index=0
  for rw in words:
      flag=0
      for j in range(0,len(r)):
          if rw==r[j][0]:
              flag=1

              appendFile.write(" "+r[j][1])
      if(flag==0):
          appendFile.write(" " + rw)

  appendFile.write("\n")






'''SLANG_WORDS_FILE = "data/slangdict.csv"
TEST_PARAGRAPHS_FILE = "data/test"
TEST_SLANG_WORDS_FILE = "data/testdata"
print("start")
PARAGRAPH_SEPARATION_TOKEN = "@@@@@"

def main():
    slang_parser = RuleSlangDetection(enchant.Dict("en_US"))
    para = ''
    input_file = open(TEST_SLANG_WORDS_FILE, 'r')
    for line in input_file:
        if(line.find(PARAGRAPH_SEPARATION_TOKEN)>=0):
            slang_parser.parseParagraph(para)
            para = ''
        else:
            para = para + line

    if(para is not ''):
        slang_parser.parseParagraph(para)

    input_file.close()

if __name__ == '__main__':
    main()'''
