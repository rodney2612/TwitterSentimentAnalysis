from sklearn.metrics.pairwise import cosine_similarity

word_vec_dict = {}
with open("../pte_imdb_42k.emb","rb") as vec_file:
	vec_file.readline() #ignore 1st line
	for line in vec_file:
		processed_line = line.decode('utf-8').strip()
		split_line = processed_line.split()
		remaining_list = split_line[1:]
		remaining_list = [float(i) for i in remaining_list]
		word_vec_dict[split_line[0]] = remaining_list

print(len(word_vec_dict))

analogy_file = open("analogy_vectors.txt","a")
correct = 0
no_of_lines_in_dataset = 0
with open("shuffled_reduced_analogy_dataset.txt","rb") as analogy_dataset:
	for line in analogy_dataset:
		split_line = line.decode('utf-8').strip().split()
		if(split_line[0] in word_vec_dict and split_line[1] in word_vec_dict and split_line[2] in word_vec_dict and split_line[3] in word_vec_dict):
			vec0 = word_vec_dict[split_line[0]]
			vec1 = word_vec_dict[split_line[1]]
			vec2 = word_vec_dict[split_line[2]]
			vec3 = word_vec_dict[split_line[3]]
			vec_sum = [a + b for a, b in zip(vec1,vec2)]
			vec_diff = [a - b for a, b in zip(vec_sum, vec0)]
			max_similarity = -2.0
			max_similar_word = ""
			if  no_of_lines_in_dataset != 0 and no_of_lines_in_dataset%100 == 0:
				accuracy = (correct * 1.0)/(no_of_lines_in_dataset)
				print(str(no_of_lines_in_dataset) + " " + str(accuracy))
				analogy_file.write(str(no_of_lines_in_dataset) + " " + str(accuracy))
			no_of_lines_in_dataset += 1
			for key in word_vec_dict.keys():
				cosine_sim = cosine_similarity([vec_diff],[word_vec_dict[key]])[0][0]
				if(key != split_line[0] and key != split_line[1] and key != split_line[2]):
					if cosine_sim > max_similarity:
						max_similarity = cosine_sim
						max_similar_word = key
			if(max_similar_word == split_line[3]):
				correct += 1
			print(split_line[1] + " + " + split_line[2] + " - " + split_line[0] + " = " + max_similar_word + " : " + str(max_similarity))
			analogy_file.write(split_line[1] + " + " + split_line[2] + " - " + split_line[0] + " = " + max_similar_word + " : " + str(max_similarity) + "\n")

accuracy = (correct * 1.0)/(no_of_lines_in_dataset)
print("Accuracy of embedding on word analogy task: ",accuracy)
