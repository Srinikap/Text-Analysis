# -*- coding: utf-8 -*-
"""VADERLexicon + SVC+ LR+ DT+ NB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1prY6Pu5vC89Vh9fU5mZbJFFtcyzELQe_
"""

import pandas as pd
import numpy as np
from time import time

textdf = pd.read_csv('C:/DoPython/Venv39/Scripts/Project/preprocessed_tweets.csv', sep=',')
textdf.info()
textdf

# Sentiment analysis using VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def predict_sentiment(review, printout, sentiment):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)  # scores has 4 values: neg, neu, pos, compound
    pred = 'non-negative' if scores['compound'] > 0 else 'negative'
    if (printout==True):
        print(review)
        print('neg',scores['neg'],', neu',scores['neu'],', pos',scores['pos'],', normalized_sum',scores['compound'])
        print('Predicted:', pred, ', Actual:', sentiment, '\n')
    return pred
# show results for individual reviews
for i in range(0, len(textdf.index)):
    predict_sentiment(textdf.lemmatized_text_str[i], True, textdf.airline_sentiment[i])

# Evaluation of VADER results
text_all = textdf.lemmatized_text_str
y = textdf.airline_sentiment
pred_list = [predict_sentiment(review, printout=False, sentiment=y[0]) for review in text_all]
predicted = pd.DataFrame(pred_list, columns=['pred'])

from sklearn import metrics
from sklearn.metrics import confusion_matrix
print('Evaluation results:\n' + 'Accuracy:', metrics.accuracy_score(y, predicted))
cm = pd.DataFrame(confusion_matrix(y, predicted, labels=['non-negative', 'negative']),
                  index=['actual non-negative', 'actual negative'],
                  columns=['predicted non-negative', 'predicted negative'])
print('Confusion matrix:\n', cm)
TP = cm.at['actual non-negative', 'predicted non-negative']
FP = cm.at['actual negative', 'predicted non-negative']
FN = cm.at['actual non-negative', 'predicted negative']
TN = cm.at['actual negative', 'predicted negative']
print('precision =', TP/(TP+FP), ', recall =', TP/(TP+FN), ', F1-score =', 2*TP/(2*TP+FP+FN), ', Specificity =', TN/(TN+FP))

"""# TFIDF matric with Lemmatization"""

# Compute term-document (tf) matrix and TFIDF matrix with lemmatization
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

def lemma_tokenizer(corpus):   # a method to lemmatize corpus
    corpus = ''.join([ch for ch in corpus if ch not in string.punctuation])  # remove punctuation
    tokens = nltk.word_tokenize(corpus)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# use nltk stopwords list
nltk_stopwords = nltk.corpus.stopwords.words('english')
print(nltk_stopwords)

"""# Split data for holdout test"""

# Split data for holdout test
from sklearn.model_selection import train_test_split
text_train, text_test, y_train, y_test = train_test_split(text_all, y, test_size=0.2, random_state=12)

# Compute term-document (tf) matrix with lemmatization
tf = CountVectorizer(tokenizer=lemma_tokenizer,  # use lemma_tokenizer
                     stop_words=nltk_stopwords,  # use customized stopwords list
                     ngram_range=(1,2))          # use unigrams and bigrams
tf_train = tf.fit_transform(text_train)
tf_train

print(text_test, '\n', y_test)
tf_test = tf.transform(text_test)
tf_test

# term-document (tf) matrix for the entire corpus - for cv test
tf_all = tf.fit_transform(text_all)
tf_all

# Compute tfidf matrix with lemmatization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(tokenizer=lemma_tokenizer,  # use lemma_tokenizer
                        stop_words=nltk_stopwords,  # use customized stopwords list
                        ngram_range=(1,2))          # use unigrams and bigrams
tfidf_train = tfidf.fit_transform(text_train)
tfidf_train

tfidf_test = tfidf.transform(text_test)
tfidf_test

# tfidf matrix for the entire corpus - for cv test
tfidf_all = tfidf.fit_transform(text_all)
tfidf_all

# Method to perform holdout test
def holdout_test(model, X_train, y_train, X_test, y_test, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(title, '\nAccuracy:', metrics.accuracy_score(y_test, y_pred))
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=['non-negative', 'negative']),
                      index=['actual non-negative', 'actual negative'],
                      columns=['predicted non-negative', 'predicted negative'])
    print('Confusion matrix:\n', cm)
    TP = cm.at['actual non-negative', 'predicted non-negative']
    FP = cm.at['actual negative', 'predicted non-negative']
    FN = cm.at['actual non-negative', 'predicted negative']
    TN = cm.at['actual negative', 'predicted negative']
    print('precision =', TP/(TP+FP), ', recall =', TP/(TP+FN), ', F1-score =', 2*TP/(2*TP+FP+FN), ', specificity =', TN/(TN+FP))

# Method to perform cross-validation test using for loop
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

def cv_test(model, X_sparse, y, title):  # X_sparse is a scipy.sparse.csr_matrix (tf or tfidf matrix)
    num_total_tested = 0
    num_correctly_classified = 0   # to calculate average accuracy over k test sets
    cm_sum = np.zeros((2,2)) # initialize a 2x2 confusion matrix (cm) for summing up the cm's from all folds
    for train_index, test_index in kf.split(X_sparse, y):
        X_train, X_test = tf_all[train_index], tf_all[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print('X_train.shape:', X_train.shape, '\ny_train.index:', y_train.index)
        # print('X_test.shape:', X_test.shape, '\ny_test.index:', y_test.index)
        # print('X_train:\n', X_train, '\ny_train\n', y_train)
        # print('X_test:\n', X_test, '\ny_test\n', y_test)
        model.fit(X_train, y_train)
        num_total_tested += len(y_test)   # num_total_tested = num_total_tested + len(y_test)
        num_correctly_classified += metrics.accuracy_score(y_test, model.predict(X_test), normalize=False)
        # print(num_total_tested, num_correctly_classified)
        cm = pd.DataFrame(confusion_matrix(y_test, model.predict(X_test), labels=['non-negative', 'negative']), 
                          index=['actual non-negative', 'actual negative'], 
                          columns=['predicted non-negative', 'predicted negative'])
        # print(cm)
        cm_sum += cm

    print(title, '\nAverage accuracy:', num_correctly_classified/num_total_tested)
    print('Confusion matrix:\n', cm_sum)
    print("'Negative' is the Target Class")
    TP = cm.at['actual non-negative', 'predicted non-negative']
    FP = cm.at['actual negative', 'predicted non-negative']
    FN = cm.at['actual non-negative', 'predicted negative']
    TN = cm.at['actual negative', 'predicted negative']
    print('precision =', TP/(TP+FP), ', recall =', TP/(TP+FN), ', F1-score =', 2*TP/(2*TP+FP+FN), ', specificity =', TN/(TN+FP))


# Method to perform cross-validation test using cross_val_score - results are inaccurate

# To get precision, recall and F1 scores from cross_val_score(), y values must be labeled {0,1}
# when cross_val_score() computes these scores, it always considers 1 as 'positive' (and 0 as 'negative')
y01 = y.apply(lambda x: 1 if x=='non-negative' else 0) # the scores are calculated by considering '+' as 'Non-Negative'
# y01 = y.apply(lambda x: 1 if x=='negative' else 0) # the scores are calculated by considering '-' as 'Negative'

from sklearn.model_selection import cross_val_score
n_folds = 5
def cv_test_inaccurate(model, X_sparse, y01, title):
    print(title)
    accuracy = cross_val_score(model, X_sparse, y01, cv=n_folds, scoring='accuracy')
    # print('Accuracy for each fold:', accuracy)
    print('Average accuracy:', accuracy.mean())
    precision = cross_val_score(model, X_sparse, y01, cv=n_folds, scoring='precision')
    # print('Precision:', precision)
    print('Average precision:', precision.mean())
    recall = cross_val_score(model, X_sparse, y01, cv=n_folds, scoring='recall')
    # print('Recall:', recall)
    print('Average recall:', precision.mean())
    f1 = cross_val_score(model, X_sparse, y01, cv=n_folds, scoring='f1')
    # print('F-1 score:',f1)
    print('Average F-1 score:', f1.mean())

"""# Naive Bayes"""

# Sentiment analysis using Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
holdout_test(mnb, tf_train, y_train, tf_test, y_test, 'Naive Bayes holdout test with tf matrix:')
holdout_test(mnb, tfidf_train, y_train, tfidf_test, y_test, '\nNaive Bayes holdout test with tfidf matrix:')

"""# Logistic Regression"""

# Sentiment analysis using logistic regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=1)
holdout_test(logit, tf_train, y_train, tf_test, y_test, 'Logistic regression holdout test with tf matrix:')
holdout_test(logit, tfidf_train, y_train, tfidf_test, y_test, '\nLogistic reg holdout test with tfidf matrix:')

"""# Support Vector Classifier SVC"""

# Sentiment analysis using support vector classifier (SVC)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
holdout_test(svc, tf_train, y_train, tf_test, y_test, 'SVC holdout test with tf matrix:')
holdout_test(svc, tfidf_train, y_train, tfidf_test, y_test, '\nSVC holdout test with tfidf matrix:')

"""# Decision Tree"""

# Sentiment analysis using decision trees
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=1)
holdout_test(tree, tf_train, y_train, tf_test, y_test, 'Decision tree holdout test with tf matrix:')
holdout_test(tree, tfidf_train, y_train, tfidf_test, y_test, '\nDecision tree holdout test with tfidf matrix:')