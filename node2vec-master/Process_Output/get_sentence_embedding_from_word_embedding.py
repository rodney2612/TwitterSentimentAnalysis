
word_vec_dict = {}
with open("node2vec_word.vec","rb") as vec_file:
	vec_file.readline() #ignore 1st line
	for line in vec_file:
		processed_line = line.decode('utf-8').strip()
		split_line = processed_line.split()
		remaining_list = split_line[1:]
		remaining_list = [float(i) for i in remaining_list]
		word_vec_dict[split_line[0]] = remaining_list

print(len(word_vec_dict))

no_of_dimesions = 100
doc_vec_file = open("node2vec_doc.vec","a")
no_of_lines = sum(1 for line in open('text_train.txt'))
doc_vec_file.write(str(no_of_lines) + " " + str(no_of_dimesions) + "\n")
with open("text_train.txt","rb") as text_file:
	for line in text_file:
		split_line = line.decode('utf-8').strip().split()
		word_vecs = []
		for i in split_line:
			word_vecs.append(word_vec_dict[i])
		doc_vec = [sum(x)/len(line) for x in zip(*word_vecs)]
		print(" ".join(str(x) for x in doc_vec) + "\n")
		doc_vec_file.write(" ".join(str(x) for x in doc_vec) + "\n")