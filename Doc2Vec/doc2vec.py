from gensim.models import Doc2Vec
import gensim
import os
import collections
import smart_open
import random

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_file_name = "text_all.txt"
train_file = open(train_file_name,'rb')
#train_corpus = list(read_corpus(train_file))

train_corpus = []
i = 0
with open(train_file_name,'rb') as file:
	for line in file:
		train_corpus.append(gensim.models.doc2vec.TaggedDocument(words=line.decode('utf-8').split(), tags=[i]))
		i += 1

# check_file = open("check_file.txt",'a')
# for i in range(42000):
# 	print(train_corpus[i])
print(len(train_corpus))
model_dir = "SavedModel/"
model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40,dm=1)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
model.save(model_dir + 'model.bin')
model.save_word2vec_format(model_dir + 'doc.vec', doctag_vec=True, word_vec=False, fvocab=None, binary=False)
model.save_word2vec_format(model_dir + 'word.vec', doctag_vec=False, word_vec=True, fvocab=None, binary=False)
new_model = Doc2Vec.load(model_dir + 'model.bin')

#print(new_model)
