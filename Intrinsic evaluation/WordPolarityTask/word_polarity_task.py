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

dataset_dir = "opinion-lexicon-English/"
pos_file_name = dataset_dir + "positive-words.txt"
neg_file_name = dataset_dir + "negative-words.txt"
pos_words = []
with open(pos_file_name,"rb") as pos_file:
	lines = pos_file.readlines()
	pos_words = [i.decode('utf-8').strip() for i in lines]
	#print(pos_words)

neg_words = []
with open(neg_file_name,"rb") as neg_file:
	lines = neg_file.readlines()
	neg_words = [i.decode('utf-8').strip() for i in lines]
	#print(neg_words)

same_polarity_array = []
opposite_polarity_array = []
no_of_pos_words = len(pos_words)
pos_pos_file = open("pos_pos_file.txt","a")
for i in range(no_of_pos_words):
	for j in range(i+1,no_of_pos_words):
		if pos_words[i] in word_vec_dict and pos_words[j] in word_vec_dict:
			vec1 = word_vec_dict[pos_words[i]]
			vec2 = word_vec_dict[pos_words[j]]
			similarity = cosine_similarity([vec1],[vec2])[0][0]
			same_polarity_array.append(similarity)
			pos_pos_file.write(pos_words[i] + " " + pos_words[j] + " " + str(similarity) + "\n")
			#print(pos_words[i] + " " + pos_words[j] + " " + str(cosine_similarity([vec1],[vec2])))

print("Pos pos combinations done")

no_of_neg_words = len(neg_words) 
pos_neg_file = open("pos_neg_file.txt","a")
for i in range(no_of_pos_words):
	for j in range(no_of_neg_words):
		if pos_words[i] in word_vec_dict and neg_words[j] in word_vec_dict:
			vec1 = word_vec_dict[pos_words[i]]
			vec2 = word_vec_dict[neg_words[j]]
			similarity = cosine_similarity([vec1],[vec2])[0][0]
			opposite_polarity_array.append(similarity)
			pos_neg_file.write(pos_words[i] + " " + neg_words[j] + " " + str(similarity) + "\n")
			#print(pos_words[i] + " " + pos_words[j] + " " + str(cosine_similarity([vec1],[vec2])))

print("Pos neg combinations done")

neg_neg_file = open("neg_neg_file.txt","a")
for i in range(no_of_neg_words):
	for j in range(i+1,no_of_neg_words):
		if neg_words[i] in word_vec_dict and neg_words[j] in word_vec_dict:
			vec1 = word_vec_dict[neg_words[i]]
			vec2 = word_vec_dict[neg_words[j]]
			similarity = cosine_similarity([vec1],[vec2])[0][0]
			same_polarity_array.append(similarity)
			#print(neg_words[i] + " " + neg_words[j] + " " + str(similarity) + "\n")
			neg_neg_file.write(neg_words[i] + " " + neg_words[j] + " " + str(similarity) + "\n")
print("Neg neg combinations done")

n1 = 0
n2 = 0
# n = len(opposite_polarity_array)
# p = len(same_polarity_array)
# for same_polarity_value in same_polarity_array:
# 	for opposite_polarity_value in opposite_polarity_array:
# 		if same_polarity_value > opposite_polarity_value:
# 			n1 += 1
# 		if same_polarity_value == opposite_polarity_value:
# 			n2 += 1

#Took long time to run. So, we have divided by 30
n = int(len(opposite_polarity_array)/30)
p = int(len(same_polarity_array)/30)
print("n: ",n)
print("p: ",p)
for i in range(p):
	#print(i)
	for j in range(n):
		if same_polarity_array[i] > opposite_polarity_array[j]:
			n1 += 1
		if same_polarity_array[i] == opposite_polarity_array[j]:
			n2 += 1


auc = (n1 + 0.5 * n2)/(n * p)

print("AUC of embedding on word polarity task: ",auc)
