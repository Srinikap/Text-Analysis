#!/usr/bin/env python
# coding: utf-8

# ## Data Preprocessing

# In[1]:


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
tweetsdf = pd.read_csv(r"C:\Users\srini\OneDrive\Desktop\Advanced Machine Learning\Project\Raw data file.csv")
tweetsdf.info()


# In[2]:


tweetsdf.head()


# In[3]:


# Drop columns
tweetsdf.drop(columns = ['tweet_id', 'airline_sentiment_gold', 'negativereason_gold', 'tweet_coord', 'tweet_location', 'airline_sentiment_confidence', 'negativereason_confidence', 'name', 'retweet_count', 'tweet_created', 'user_timezone', 'negativereason', 'airline'], axis = 1, inplace = True)


# In[4]:


# NUmber of each sentiment reviews
print('Number of negative tweets:', tweetsdf[tweetsdf['airline_sentiment'] == 'negative']['airline_sentiment'].count())
print('Number of positive tweets:', tweetsdf[tweetsdf['airline_sentiment'] == 'positive']['airline_sentiment'].count())
print('Number of neutral tweets:', tweetsdf[tweetsdf['airline_sentiment'] == 'neutral']['airline_sentiment'].count())


# In[5]:


import seaborn as sns
sns.countplot(x = "airline_sentiment", data = tweetsdf)


# In[6]:


# Replacing 'neutral' & 'positive' with 'non-negative' respectively
tweetsdf['airline_sentiment'].replace('positive', 'non-negative', inplace=True)
tweetsdf['airline_sentiment'].replace('neutral', 'non-negative', inplace=True)
tweetsdf.head()


# In[7]:


# Finding the duplicate values
tweetsdf.duplicated().sum()


# In[8]:


# Dropping duplicates
tweetsdf = tweetsdf.drop_duplicates(keep='first')
tweetsdf.duplicated().sum()


# In[9]:


# Checking for any null values
tweetsdf.isnull().all()


# In[10]:


tweetsdf.head()


# In[11]:


tweetsdf.info()


# In[12]:


import seaborn as sns
sns.countplot(x = "airline_sentiment", data = tweetsdf)


# In[13]:


tweetsdf.text[73]


# In[14]:


tweetsdf.text[233]


# In[15]:


tweetsdf.text[24]


# In[16]:


# Remove any URL's from the data
import re
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

tweetsdf['no_URL'] = tweetsdf['text'].apply(lambda x: [remove_URL(word) for word in x.split()])
tweetsdf.head()


# In[17]:


tweetsdf.no_URL[73]


# In[18]:


# Remove any html tags from the data
def strip_html_tags(text):   
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

tweetsdf['no_html'] = tweetsdf['no_URL'].apply(lambda x: [strip_html_tags(word) for word in x])
tweetsdf.head()


# In[19]:


# Remove accented characters
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text

tweetsdf['no_accentchar'] = tweetsdf['no_html'].apply(lambda x: [unidecode.unidecode(word) for word in x])
tweetsdf.head()


# In[21]:


tweetsdf.no_accentchar[24]


# In[22]:


tweetsdf.no_accentchar[18]


# In[23]:


# Expanding contractions 'you've to you have'
get_ipython().system('pip install contractions')
import contractions
def expand_contractions(text):
    text = contractions.fix(text)
    return text


# In[25]:


tweetsdf['no_contract'] = tweetsdf['no_accentchar'].apply(lambda x: [contractions.fix(word) for word in x])
tweetsdf.head()


# In[26]:


tweetsdf['no_contract_str'] = [' '.join(map(str, l)) for l in tweetsdf['no_contract']]
tweetsdf.head()


# In[27]:


# Remove punctuations
import string
def remove_punct(text):
    text_nonpunct = "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

tweetsdf['no_punc_text'] = tweetsdf['no_contract_str'].apply(lambda x: remove_punct(x))
tweetsdf.head()


# In[29]:


tweetsdf.no_punc_text[18]


# In[30]:


# Tokenization of the data
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
tweetsdf['tokenized'] = tweetsdf['no_punc_text'].apply(word_tokenize)
tweetsdf.head()


# In[31]:


tweetsdf.tokenized[18]


# In[32]:


# convert text to lower case
tweetsdf['lower'] = tweetsdf['tokenized'].apply(lambda x: [word.lower() for word in x])
tweetsdf.head()


# In[33]:


tweetsdf.lower[18]


# In[36]:


# remove stop words
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.remove('but')
nltk_stopwords.remove('no')
nltk_stopwords.remove('not')

stop_words = set(stopwords.words('english'))
tweetsdf['no_stopwords'] = tweetsdf['lower'].apply(lambda x: [word for word in x if word not in stop_words])
tweetsdf.head()


# In[38]:


tweetsdf.no_stopwords[18]


# In[39]:


tweetsdf.no_stopwords[328]


# In[40]:


# parts of speech of each word for lemmatization purpose
nltk.download('averaged_perceptron_tagger')
tweetsdf['pos_tags'] = tweetsdf['no_stopwords'].apply(nltk.tag.pos_tag)
tweetsdf.head()


# In[41]:


tweetsdf.pos_tags[18]


# In[42]:


nltk.download('wordnet')
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
tweetsdf['wordnet_pos'] = tweetsdf['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
tweetsdf.head()


# In[43]:


tweetsdf.wordnet_pos[18]


# In[44]:


# Lemmatization of data
wnl = WordNetLemmatizer()
tweetsdf['lemmatized'] = tweetsdf['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
tweetsdf.head()


# In[45]:


tweetsdf.lemmatized[18]


# In[46]:


tweetsdf['lemmatized_str'] = [' '.join(map(str, l)) for l in tweetsdf['lemmatized']]
tweetsdf.head()


# In[47]:


tweetsdf.lemmatized_str[18]


# In[49]:


# Drop columns
tweetsdf.drop(columns = ['text', 'no_URL', 'no_html', 'no_accentchar', 'no_contract', 'no_contract_str', 'no_punc_text', 'tokenized', 'lower', 'no_stopwords', 'pos_tags', 'wordnet_pos', 'lemmatized'], axis = 1, inplace = True)


# In[50]:


# Saving the preprocessed text file tin csv format
tweetsdf.to_csv('Finaldf1.csv')


# In[32]:


tweetsdf.head()

