from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

word_vec_dict = {}
with open("../doc2vec_word.vec","rb") as vec_file:
	vec_file.readline() #ignore 1st line
	for line in vec_file:
		processed_line = line.decode('utf-8').strip()
		split_line = processed_line.split()
		remaining_list = split_line[1:]
		remaining_list = [float(i) for i in remaining_list]
		word_vec_dict[split_line[0]] = remaining_list

print("Total no of unique words in dataset: ",len(word_vec_dict))


'''
Relatedness:
These  datasets  contain  relatedness scores for pairs of words;  
the cosine similarity of the embeddings for two words should  have  
high  correlation  (Spearman  or Pearson) with human relatedness scores
'''

#do evaluation using wordsim353 dataset - word similarity dataset

similarity_file = open("similarity_results.txt","a")
human_array = []
similarity_array = []
human = []
simi = []
i = 0
with open("combined.tab","rb") as similarity_dataset:
	for line in similarity_dataset:
		split_line = line.decode('utf-8').split('\t')
		if(split_line[0] in word_vec_dict and split_line[1] in word_vec_dict):
			vec1 = word_vec_dict[split_line[0]]
			vec2 = word_vec_dict[split_line[1]]
			i += 1
			human_array.append([i, float(split_line[2])])
			human.append(float(split_line[2]))
			similarity = cosine_similarity([vec1],[vec2])[0][0] * 10
			simi.append(similarity)
			similarity_array.append([i, similarity])
			#print(split_line[0] + " " + split_line[1] + " " + split_line[2].strip() + " " + str(similarity))
			similarity_file.write(split_line[0] + " " + split_line[1] + " " + split_line[2].strip() + " " + str(similarity) + "\n")

human_array = sorted(human_array,key=lambda x: (x[1]),reverse = True)
similarity_array = sorted(similarity_array,key=lambda x: (x[1]),reverse = True)

human_list = [row[0] for row in human_array]
similarity_list = [row[0] for row in similarity_array]

#correlation,t = spearmanr(human_list,similarity_list)
#print("Correlation between human and embedding similarity: ",correlation)

correlation,t = spearmanr(human,simi)
print("Correlation between human and embedding similarity:",correlation)
