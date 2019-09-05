def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    :param mess: string
    :return: return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

import nltk
import pandas as pd

#nltk.download_shell()

# messages = [line.rstrip() for line in open("E:/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection")]
# print(len(messages))
# for mess_no,mess in enumerate(messages[:10]):
#     print(str(mess_no) + " " + mess)

messages = pd.read_csv("E:/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/smsspamcollection/SMSSpamCollection",sep="\t",names=["label","message"])
print(messages.head())
print()

print(messages.describe())
print()

print(messages.groupby("label").describe())
print()

messages["length"] = messages["message"].apply(len)
print(messages.head())
print()

import seaborn as sns
import matplotlib.pyplot as plt

messages["length"].hist(bins=150)
print(messages["length"].describe())
print()
messages.hist(column="length",by="label",bins=70,figsize=(12,4))

import string

from nltk.corpus import stopwords
print(stopwords.words("english"))
print()

# Tokenization and defining word vectors
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])
print(len(bow_transformer.vocabulary_))
print()

# Taking one message and checking its word vector
mess4 = messages["message"][3]
bow4 = bow_transformer.transform([mess4])
print(bow4)
print(bow4.shape)
print()

# Getting a particular word from bag of words
print(bow_transformer.get_feature_names()[4068])

# Transforming messages data frame into bag of words vectorisation
messages_bow = bow_transformer.transform(messages["message"])
print("Shape of Sparse Matrix : " + str(messages_bow.shape))
print("Non zero values : " + str(messages_bow.nnz))
sparsity = ((messages_bow.nnz/(messages_bow.shape[0]*messages_bow.shape[1]))*100)
print("Sparsity : " + str(sparsity))
print()

# Calculating tf-idf using scikit transformers
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Checking tf-idf of bow4 message
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
print()

# We can also check idf of a particular word as
print(tfidf_transformer.idf_[bow_transformer.vocabulary_["university"]])
print()

# Converting our whole bag of words corpus into tf-idf
messages_tfidf = tfidf_transformer.transform(messages_bow)


# Using Naive-Bayes Classifier to classify spam and ham messages
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages["label"])
all_prediction = spam_detect_model.predict(messages_tfidf)
print(all_prediction)
print()

# Checking prediction on single message
print(spam_detect_model.predict(tfidf4)[0])
print()

# Splitting data into training and testing data
from sklearn.model_selection import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(messages["message"],messages["label"],test_size=0.3)


# Creating pipeline of transforms
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("bow_step",CountVectorizer(analyzer=text_process)),
    ("tfidf_step",TfidfTransformer()),
    ("classifier_step",MultinomialNB())
])# our step name can be anything
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

# Calculating classification report
from sklearn.metrics import classification_report
print(classification_report(label_test,predictions))

plt.show()