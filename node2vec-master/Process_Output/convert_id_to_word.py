import pickle

pkl_file = open('node_to_word_mapping.pkl', 'rb')
node_to_word_mapping = pickle.load(pkl_file)
pkl_file.close()

#print(node_to_word_mapping)


word_vec_file = open('node2vec_word.vec','a')
with open('node2vec_id.vec','rb') as id_vec_file:
	word_vec_file.write(id_vec_file.readline().decode('utf-8'))
	for line in id_vec_file:
		split_line = line.decode('utf-8').strip().split(' ',1)
		mapped_word = node_to_word_mapping[int(split_line[0])]
		#print(mapped_word + " " + split_line[1])
		word_vec_file.write(mapped_word + " " + split_line[1] + "\n")
