import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle

neg_file_name = "42K_neg"
neu_file_name = "42K_neu"
pos_file_name = "42K_pos"
labels = {"neg":"0","neu":"1","pos":"2"}
neg_file_array = []
with open(neg_file_name,"r") as neg_file:
	for line in neg_file:
		neg_file_array.append(line.strip() + "\t" + labels["neg"] + "\n")

print(len(neg_file_array))

neu_file_array = []
with open(neu_file_name,"r") as neu_file:
	for line in neu_file:
		neu_file_array.append(line.strip() + "\t" + labels["neu"] + "\n")

print(len(neu_file_array))

pos_file_array = []
with open(pos_file_name,"r") as pos_file:
	for line in pos_file:
		pos_file_array.append(line.strip() + "\t" + labels["pos"] + "\n")

print(len(pos_file_array))

#neg_file_array = shuffle(neg_file_array)
#neu_file_array = shuffle(neu_file_array)
#pos_file_array = shuffle(pos_file_array)

train_test_partitions = [1/2.0,1/3.0]
for i in range(len(train_test_partitions)):
	train_file = open("train_" + str(i) + ".txt","a")
	test_file = open("test_" + str(i) + ".txt","a")
	file_length = len(neg_file_array)
	for j in range(file_length):
		if(j <= train_test_partitions[i] * file_length):
			train_file.write(neg_file_array[j])
		else:
			test_file.write(neg_file_array[j])
	file_length = len(neu_file_array)
	for j in range(file_length):
		if(j <= train_test_partitions[i] * file_length):
			train_file.write(neu_file_array[j])
		else:
			test_file.write(neu_file_array[j])
	file_length = len(pos_file_array)
	for j in range(file_length):
		if(j <= train_test_partitions[i] * file_length):
			train_file.write(pos_file_array[j])
		else:
			test_file.write(pos_file_array[j])
