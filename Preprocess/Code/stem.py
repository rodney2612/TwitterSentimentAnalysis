from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

out = open("stemmed_text_all.txt","a")
line_no = 0
with open("text_all.txt","rb") as infile:
	for line in infile:
		split_line = line.decode('utf-8').strip().split()
		stemmed_line = ""
		for word in split_line:
			stemmed_line += ps.stem(word) + " "
		stemmed_line.strip()
		line_no += 1
		print(line_no)
		out.write(stemmed_line + "\n")
