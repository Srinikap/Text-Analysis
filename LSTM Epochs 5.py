#!/usr/bin/env python
# coding: utf-8

# ## LSTM with epochs = 5 & Vec size 10000

# In[2]:


# Package imports
import pandas as pd
import numpy as np
from time import time
from bs4 import BeautifulSoup
import spacy
get_ipython().system('pip install unidecode')
get_ipython().system('pip install word2number')
import unidecode
from word2number import w2n
get_ipython().system('pip install contractions')
import contractions

# Loading the data into pandas dataframe
tweetsdf = pd.read_csv(r"C:\Users\srini\OneDrive\Desktop\Advanced Machine Learning\Project\Preprocessed data file.csv")
tweetsdf.info()


# In[3]:


tweetsdf.drop(columns = ['Unnamed: 0' ], axis = 1, inplace = True)


# In[5]:


# method to tokenize documents with lemmatization
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

def lemma_tokenizer(corpus):   # a method to lemmatize corpus
    corpus = ''.join([ch for ch in corpus if ch not in string.punctuation])  # remove punctuation
    tokens = nltk.word_tokenize(corpus)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

nltk_stopwords = nltk.corpus.stopwords.words('english')

# construct term-document (tf) matrix with lemmatization - not using data yet
tf = CountVectorizer(tokenizer=lemma_tokenizer,  # use lemma_tokenizer
                     stop_words=nltk_stopwords,  # use customized stopwords list
                     ngram_range=(1,2))          # use unigrams and bigrams


# In[6]:


# Split the data
from sklearn.model_selection import train_test_split
textdf = tweetsdf['lemmatized_str']
ydf = tweetsdf['airline_sentiment']
textdf_train, textdf_test, ydf_train, ydf_test = train_test_split(textdf, ydf, test_size=0.2, random_state=12)
textdf_test[:5], ydf_test[:5]


# In[7]:


tf_train = tf.fit_transform(textdf_train)
tf_train


# In[8]:


tf_test = tf.transform(textdf_test)
tf_test


# In[9]:


# Sentiment analysis using LSTM with supervised word embeddings

# encode label (sentiment) into a numpy arry with 0/1 values (in alphabetical order)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(ydf)
y = label_encoder.transform(ydf)
print('Encoded class order:', label_encoder.classes_)
print('Before encoding:', label_encoder.inverse_transform(y)[0:5])
print('After encoding: ', y[0:5])     # 0 = +, 1 = -

# encode Sentiment into a dataframe with 0/1 values
# y = ydf.apply(lambda x: 1 if x=='+' else 0)  # 0 = -, 1 = +
# y.head()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(textdf, y, test_size=0.2, random_state=12)
X_test[:5], y_test[:5]


# In[11]:


# tokenize documents directly using tf.keras.preprocessing.text.Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()   # tokenizer = Tokenizer(num_words=100)  # to use the 100 most frequent words
tokenizer.fit_on_texts(X_train)  # tokenize the documents and index the words


# In[12]:


# transform words in documents to sequences of indexes, required by tf.keras.layers.Embedding
print('X_train in text:\n', X_train[:5])
X_train = tokenizer.texts_to_sequences(X_train)
print('\nX_train after indexing:\n', X_train[:5])


# In[13]:


# tf.keras.layers.Embedding requires inputs to have equal length; so find max_len for this use
max_len = np.max([len(doc) for doc in X_train])
max_len


# In[14]:


# pad shorter documents with zeros so that all documents have max_len
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen=max_len)  # padding='pre' (default) or 'post'
print('X_train after padding:\n', X_train)


# In[15]:


print('X_test in text:\n', X_test[:5])
X_test = tokenizer.texts_to_sequences(X_test)
print('\nX_test after indexing:\n', X_test[:5])


# In[16]:


X_test = pad_sequences(X_test, maxlen=max_len)
print('X_test after padding:\n', X_test)


# In[17]:


vocab_size = len(tokenizer.word_index) + 1
vocab_size


# In[17]:


# design LSTM model with Embedding layer (which must be the 1st layer)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Dropout
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

vec_size = 10000   # dimensionality of the word vectors

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=vec_size, input_length=max_len))
model.add(layers.LSTM(32, return_sequences=False))
model.add(Dense(1))


model.summary()


# In[18]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.optimizer.get_config()


# In[19]:


# train LSTM model
from time import time
t0 = time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=5) 
print("\nTime to train LSTM model: %0.3f seconds." % (time() - t0))


# In[20]:


# plot training process
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# plot for loss (mse)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (Binary Cross-Entropy)', fontsize=16)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss (binary cross-entropy)', fontsize=14)
plt.legend(['train', 'valid'], loc='upper right')
plt.show()

# plot for metrics
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy', fontsize=16)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


# In[21]:


# Method to perform evaluation for LSTM on test data
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, y_test, y_pred, first_label, second_label, print_predict):

    if (print_predict==True):
        print('Prediction results:')
        print('Actual:   ', np.array(y_test))
        print('Predicted:', y_pred)

    print('\nTest accuracy:', accuracy_score(y_test, y_pred))
    print('\nConfusion matrix:\n', confusion_matrix(y_test, y_pred))
    
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[first_label, second_label]),
                      index=['actual ' + first_label, 'actual ' + second_label],
                      columns=['predicted ' + first_label, 'predicted ' + second_label])
    print('\nConfusion matrix with labels:\n', cm)
    print('\nLabel ' + first_label + ' is positive class')
    TP = cm.at['actual ' + first_label,  'predicted ' + first_label]
    FP = cm.at['actual ' + second_label, 'predicted ' + first_label]
    FN = cm.at['actual ' + first_label,  'predicted ' + second_label]
    TN = cm.at['actual ' + second_label, 'predicted ' + second_label]
    print('precision =', TP/(TP+FP), ', recall =', TP/(TP+FN), ', F1-score =', 2*TP/(2*TP+FP+FN), 'Specificity =', TN/(TN+FP))


# In[22]:


y_pred_prob = model.predict(X_test)
# y_pred = (y_pred_prob > 0.5).astype('int32')    # if y_pred_prob>0.5 y_pred=1 else y_pred=0
y_pred_prob = pd.DataFrame(y_pred_prob, columns=['y_prob'])
y_pred = y_pred_prob['y_prob'].apply(lambda x: 'negative' if x < 0.5 else 'non-negative')  # label encode 0='+', 1='-'
y_test = label_encoder.inverse_transform(y_test)
evaluate(model, y_test, np.array(y_pred), 'negative', 'non-negative', True)

