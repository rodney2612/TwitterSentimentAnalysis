import numpy as np
import random
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def file_read():
    #reading text data
    with open('combined.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    #reading label data
    with open('labels.txt') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    #randomly selects 10000 indices from
    var = random.sample(range(41960), 10000)

    #combining data with labels separated by \t
    new_path = 'text_labels.txt'
    new_vectors = open(new_path,'w')
    for i in range(10000):
        x = var[i]
        new_vectors.write(str(content[x])+"\t"+str(labels[x])+"\n")


def retrive_data():
    tweets = []
    labels = []
    data_file = "text_labels.txt"

    #retrive of data then separate into two parts one as text other as lable
    with open(data_file, "r") as ins:
        for line in ins:
            text = line.split("\t")
            tweets.append(text[0])
            labels.append(text[1])

    print(tweets[1],labels[1])

    tweet_file = "min_tweets.txt"
    label_file = "min_labels.txt"

    #writing text data in one file and label data into other
    new_tweet = open(tweet_file,"w")
    new_label = open(label_file,"w")

    for i in range(len(tweets)):
        new_tweet.write(tweets[i]+"\n")
        new_label.write(labels[i]+"\n")


def labels_all():
    label_file = "label_all.txt"
    new_label = open(label_file,"w")

    for i in range(1,178134):
        new_label.write("1"+"\n")




#Word-word embedding making
def w_w_making():
    w_w_file = open("word_emb.txt","w")
    w_w_edgelist = open("ww_edgelist.txt","w")
    ww_file_path = "ww_emb.txt"
    ww_weight = open("ww_weight_edgelist.txt","w")
    unique_words = {}
    i=1
    with open(ww_file_path, "r") as ins:
        for line in ins:
            nodes = line.split() #splitting given file
            vertex1 = nodes[0]
            vertex2 = nodes[1]
            vertex3 = nodes[2]
            print(nodes)
            # unique_words.append(vertex1)
            # unique_words.append(vertex2)
            if vertex1 not in unique_words:
                unique_words[vertex1] = i
                i+=1
            if vertex2 not in unique_words:
                unique_words[vertex2] = i
                i+=1

            # print(vertex1+" "+str(unique_words[vertex1]))
            # print(vertex2+" "+str(unique_words[vertex2]))
            w_w_file.write(vertex1+"\t"+vertex2+"\n")
            w_w_edgelist.write(str(unique_words[vertex1])+"\t"+str(unique_words[vertex2])+"\n")
            ww_weight.write(str(unique_words[vertex1])+"\t"+str(unique_words[vertex2])+"\t"+vertex3+"\n")





def find_svm(x_train,y_train,x_test,y_test):
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_prediction = clf.predict(x_test)
    #print(predi)

    cnf_matrix = confusion_matrix(y_test, y_prediction)
    print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PRECISION = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    # AUC = metrics.auc(FPR, TPR)

    return ACC, PRECISION, TPR




#SVM Computation
def svm_compute():
    sentence_vector = np.zeros([42000,100])
    sentence_labels = np.zeros([42000])
    i =0
    with open("/Users/Ashwin/Desktop/Final_docs_pte/pte_imdb_big.txt", "r") as ins:
        for line in ins:
            vector = line.split()
            for j in range(100):
                sentence_vector[i][j] = vector[j]
            i+=1
    print(sentence_vector[0])

    i=0
    with open("/Users/Ashwin/Desktop/Final_docs_pte/label_pte_imdb_big.txt", "r") as ins:
        for line in ins:
            vector = line.split()
            sentence_labels[i] = vector[0]
            i+=1
    print(sentence_labels[20000])

    kf = KFold(n_splits = 5, random_state = None, shuffle = True)

    acc=auc=prec=recall=fpr=tpr=0
    a=b=c=d=e=0
    print(a," ",b," ",c," ",d," ",e)

    for train_index, test_index in kf.split(sentence_vector):
        x_train, x_test = sentence_vector[train_index], sentence_vector[test_index]
        y_train, y_test = sentence_labels[train_index], sentence_labels[test_index]
        a, c, b = find_svm(x_train,y_train,x_test,y_test)
        acc+=a
        prec+=c
        recall+=b
        print(a," "," ",c," ",b)

    svm_acc = acc/5
    # svm_auc = auc/5
    svm_prec = prec/5
    svm_recall = recall/5
    # svm_fpr = fpr/5
    # svm_tpr = tpr/5
    f_measure = (1/svm_prec)+(1/svm_recall)

    print("acc = ",svm_acc," prec= ",svm_prec," recall= ",svm_recall," f_measure= ",f_measure)

num_max = 100
max_len = 100


sentence_vector = []
sentence_labels = []





#simple model of CNN
def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc',metrics.binary_accuracy])
    print('compile done')
    return model


#new implementation for CNN (Hope it works)
def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(130000, 100, input_length=100))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam',    metrics=['accuracy'])
    return model_conv






#Starting of CNN
def cnn_padding():

    i =0
    with open("/Users/Ashwin/Desktop/Final_docs_pte/doc2vec_imdb_text.txt", "r") as ins:
        for line in ins:
            sentence_vector.append(line)

    print(sentence_vector[14])
    i=0
    with open("/Users/Ashwin/Desktop/Final_docs_pte/label_imdb_doc2vec.txt", "r") as ins:
        for line in ins:
            sentence_labels.append(line)
    print(sentence_labels[20200])
    MAX_NB_WORDS = 10000
    tokenizer = Tokenizer (num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(str(sentence_vector))

    sequences = tokenizer.texts_to_sequences(sentence_vector)
    data = pad_sequences(sequences, maxlen=100)

    model_conv = create_conv_model()
    model_conv.fit(data, np.array(sentence_labels), validation_split=0.4, epochs = 3)






#file_read()
#retrive_data()
#w_w_making()
#labels_all()
read_sentence()
#cnn_padding()
