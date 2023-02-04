#!/usr/bin/env python
# coding: utf-8

# ## Topic Modeling with LSA

# ### Wordcloud of the tweets

# In[22]:


get_ipython().system('pip install WordCloud')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
wordcloud = WordCloud(
                          collocations = False,
                          width=1600, height=800,
                          background_color='white',
                          max_words=150,
                          #max_font_size=40, 
                          random_state=42
                         ).generate(' '.join(tweetsdf['lemmatized_str'])) # can't pass a series, needs to be strings and function computes frequencies
print(wordcloud)
plt.figure(figsize=(9,8))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[7]:


# Tokenization with sklearn and nltk: set to lower case, remove stop words, and lemmatize words
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

def lemma_tokenizer(corpus): # a method to lemmatize corpus
    corpus = ''.join([ch for ch in corpus if ch not in string.punctuation]) # remove punctuation
    tokens = nltk.word_tokenize(corpus)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

nltk_stopwords = nltk.corpus.stopwords.words('english') # use nltk's English stopwords list
tf = CountVectorizer(tokenizer=lemma_tokenizer, stop_words=nltk_stopwords) # default lowercase
tf_sparse = tf.fit_transform(tweetsdf.lemmatized_str)
tf_dictionary = tf.get_feature_names()
print(tf_dictionary)
tf_sparse


# In[8]:


tf = CountVectorizer(tokenizer=lemma_tokenizer, stop_words=nltk_stopwords) # default lowercase
tf_sparse = tf.fit_transform(tweetsdf.lemmatized_str)
tf_dictionary = tf.get_feature_names()
print(tf_dictionary)
tf_sparse


# In[9]:


tf_dense = tf_sparse.toarray() # convert sparse to dense matrix
pd.DataFrame(tf_dense, columns=tf_dictionary)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lemma_tokenizer, stop_words=nltk_stopwords) # default lowercase
tfidf_sparse = tfidf.fit_transform(tweetsdf.lemmatized_str)
tfidf_dictionary = tfidf.get_feature_names()
tfidf_sparse


# In[11]:


# sklearn for latent semantic analysis (LSA)
from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=5)
lsa


# In[12]:


lsa_tf_topics = lsa.fit_transform(tf_sparse)
lsa_tf_topics.shape


# In[13]:


# print top terms for each topic
def print_top_terms(model, vocabulary, n_top_terms):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([vocabulary[i]
                             for i in topic.argsort()[:-n_top_terms - 1:-1]])
        print(message)
    print()

print('LSA topics based on term-document matrix:')
print_top_terms(lsa, tf_dictionary, 20)


# In[80]:


Top_words_LSA = "flight united usairways get americanair cancel southwestair jetblue hour delay time help flightled service hold 2 customer u wait united bag service customer get fly plane time airline gate make would delay seat unite lose wait thank check flight cancel united flightled jetblue southwestair late attendant flighted book virginamerica tomorrow delayed problem sfo reschedule fll cancelled den bna usairways flight united hold late clt phl miss delay charlotte minute mile min philly connection dca hour delay attendant connect fly delay fleek thanks fleet time go would thank jfk delay plane great u cancel love service guy know"  


# ### Word Cloud for Top words of LSA with TF matrix

# In[81]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
wordcloud = WordCloud(
                          collocations = False,
                          width=1600, height=800,
                          background_color='white',
                          max_words=150,
                          #max_font_size=40, 
                          random_state=42
                         ).generate(' '.join([Top_words_LSA])) # can't pass a series, needs to be strings and function computes frequencies
print(wordcloud)
plt.figure(figsize=(9,8))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[23]:


lsa.fit_transform(tfidf_sparse)
print('LSA topics based on tfidf matrix:')
print_top_terms(lsa, tfidf_dictionary, 20)


# In[82]:


Top_words_LSA_TFIDF = 'flight united usairways americanair get southwestair thanks jetblue cancel hour help thank hold service time customer call delay flightled wait jetblue fleek fleet thanks thank rt great united much send fly guy love good dm awesome follow southwestair jfk response united thank thanks southwestair dm customer service follow send much bag great response bad appreciate yes airline ever fly make united flight cancel delay flightled jetblue late fleek fleet miss book problem tomorrow unite seat plane airline next gate connection southwestair thank flight cancel flightled dm follow tomorrow send love destinationdragons much rebook flighted southwest today book imaginedragons nashville swa'


# ### Word Cloud for Top words of LSA with TFIDF matrix

# In[83]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud 
wordcloud = WordCloud(
                          collocations = False,
                          width=1600, height=800,
                          background_color='white',
                          max_words=150,
                          #max_font_size=40, 
                          random_state=42
                         ).generate(' '.join([Top_words_LSA_TFIDF])) # can't pass a series, needs to be strings and function computes frequencies
print(wordcloud)
plt.figure(figsize=(9,8))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

