no_of_train_test_partitions = 2
for i in range(no_of_train_test_partitions):
	train_file_name = "train_" + str(i) + ".txt"
	train_data_file_name = "train_data_" + str(i) + ".txt"
	train_data_file = open(train_data_file_name,"a")
	train_label_file_name = "train_label_" + str(i) + ".txt"
	train_label_file = open(train_label_file_name,"a")
	with open(train_file_name,"r") as train_file:
		for line in train_file:
			data_label_split = line.split("\t")
			train_data_file.write(data_label_split[0] + "\n")
			train_label_file.write(data_label_split[1])
	train_data_file.close()
	train_label_file.close()
