# -*- coding: utf-8 -*-
"""
#TOXIC COMMENT CLASSIFIER

PROBLEM STATEMENT:

DATASOURCE: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The Toxic Comment Classifier is a machine learning (NLP) technology which classifies comments as either toxic or non-toxic. The project will rely on a dataset from Kaggle, which contains various statements that can further be classified into the following variables: toxicity, severe toxicity, threats, obscenity, insults, and identity hatred.
The size of the dataset is 159571.

STEP #1: IMPORT LIBRARIES
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings 
import pickle
warnings.filterwarnings('ignore')

# !pip install nltk
# !pip install contractions
# !pip install nlp_utils

#other libraries to help with training:
import nlp_utils
import re #for regex operations
import string
import nltk #natural language toolkit
import contractions #for expanding shortened phrases that use apostrophes
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords #for removing words whose absence won't change the sentence meaning
from nltk.tokenize import word_tokenize, sent_tokenize #to translate words or sentences into tokens


from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import ngrams

"""STEP #2: IMPORT DATASET"""

# from google.colab import drive
# drive.mount('/content/drive')

# !ls '/content/drive/My Drive/Group4_Toxic Comment Classifier/'

training_set = pd.read_csv(
  "C:/Users/HP/OneDrive - Ashesi University/L200/SECOND SEMESTER/Intro to AI/Group 4 Project/Toxic Comment Classifier/Notebook/toxic_comment_dataset.csv"
  )

# training_set.info()

#15,294 of the comments are toxic
x1 = training_set['toxic'].value_counts()

#1595 of the comments are severe toxic
x2 = training_set['severe_toxic'].value_counts()

#8449 of the comments are obscene
x3 = training_set['obscene'].value_counts()

#478 of the comments are threats
x4 = training_set['threat'].value_counts()

#7877 of the comments are insults
x5 = training_set['insult'].value_counts()

#1405 of the comments are identity-hate related
x6 = training_set['identity_hate'].value_counts()

"""STEP#3: EXPLORE/VISUALIZE DATA"""

#only numeric datatypes can be visualised
# the columns containing the comments cannot be visualised
int_columns = training_set.iloc[:,2:].sum()

# int_columns

sns.set_style("darkgrid")
sort_vals = int_columns.sort_values(ascending=False)
plt.figure(figsize=(15,8))
bar_rep = sns.barplot(sort_vals.index, sort_vals.values, alpha=0.8)
plt.title("Toxic Comments Distribution")
plt.xlabel("Various Sentence Types")
plt.ylabel("Count")
bar_rep.set_xticklabels(labels=sort_vals.index,fontsize=10)
# plt.show()

"""STEP #4: PREPARE THE DATA FOR CLEANING"""

#the sentences in the comments column contain characters other than letters
training_set['comment_text'][10]

#using regex to remove irrelelvant characters
rem_alph_num = lambda x: re.sub('\w*\d\w*', ' ', x)
text_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
rem_newline = lambda x: re.sub("\n", ' ', x)
rem_nonascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
training_set['comment_text'] = training_set['comment_text'].map(rem_alph_num).map(text_lower).map(rem_newline).map(rem_nonascii)

#creating subsets from the main dataset
# insults dataframe
insult = training_set.loc[:,['id', 'comment_text', 'insult']]

# identity hate dataframe
identity_hate = training_set.loc[:,['id', 'comment_text', 'identity_hate']]

# obscenity dataframe
obscene = training_set.loc[:,['id', 'comment_text', 'obscene']]

# threats dataframe
threat = training_set.loc[:,['id', 'comment_text', 'threat']]

# severe toxic dataframe
severe_toxic = training_set.loc[:,['id', 'comment_text', 'severe_toxic']]

# toxic dataframe
toxic = training_set.loc[:,['id', 'comment_text', 'toxic']]


#creating a word cloud of most used words in each category
def wordcloud(training_set, label):
  subset = training_set[training_set[label] == 1]
  word = subset.comment_text.values
  wcloud = WordCloud(background_color='white', max_words=2000)

  wcloud.generate(" ".join(word))
  plt.figure(figsize=(20,20))
  plt.subplot(221)
  plt.axis("off")
  plt.title("Most used words in {}".format(label), fontsize=20)
  plt.imshow(wcloud.recolor(colormap='gist_earth', random_state=244), alpha=0.98)

wordcloud(insult, 'insult')

wordcloud(identity_hate, 'identity_hate')

wordcloud(obscene, 'obscene')

wordcloud(threat, 'threat')

wordcloud(severe_toxic, 'severe_toxic')

wordcloud(toxic, 'toxic')

#ratio of variables to their negation is unbalanced
#create a balance by selecting same number of frequency both ends

#select 7877 insult comments
insult_1 = insult[insult['insult'] == 1].iloc[0:7877,:]

#select 7877 non insult comments
insult_0 = insult[insult['insult'] == 0].iloc[0:7877,:]

#concatenate insult with non-insult
insult_balanced = pd.concat([insult_1, insult_0])

insult_balanced['insult'].value_counts()

#select 1405 identity hate comments
identity_hate_1 = identity_hate[identity_hate['identity_hate'] == 1].iloc[0:1405,:]

#select 1405 identity hate comments
identity_hate_0 = identity_hate[identity_hate['identity_hate'] == 0].iloc[0:1405,:]

#concatenate identity hate with non-identity hate
identity_hate_balanced = pd.concat([identity_hate_1, identity_hate_0])

identity_hate_balanced['identity_hate'].value_counts()

#select 8449 obscene comments
obscene_1 = obscene[obscene['obscene'] == 1].iloc[0:8449,:]

#select 8449 non obscene comments
obscene_0 = obscene[obscene['obscene'] == 0].iloc[0:8449,:]

#concatenate obscene with non-obscene
obscene_balanced = pd.concat([obscene_1, obscene_0])

obscene_balanced['obscene'].value_counts()

#select 478 threat comments
threat_1 = threat[threat['threat'] == 1].iloc[0:478,:]

#select 478 threat comments
threat_0 = threat[threat['threat'] == 0].iloc[0:478,:]

#concatenate identity hate with non-identity hate
threat_balanced = pd.concat([threat_1, threat_0])

threat_balanced['threat'].value_counts()

#select 1595 severe toxic comments
severe_toxic_1 = severe_toxic[severe_toxic['severe_toxic'] == 1].iloc[0:1595,:]

#select 500 non severe toxic comments
severe_toxic_0 = severe_toxic[severe_toxic['severe_toxic'] == 0].iloc[0:1595,:]

#concatenate severe toxic with non-severe toxic
severe_toxic_balanced = pd.concat([severe_toxic_1, severe_toxic_0])

severe_toxic_balanced['severe_toxic'].value_counts()

#select 5000 toxic comments
toxic_1 = toxic[toxic['toxic'] == 1].iloc[0:5000,:]

#select 500 non toxic comments
toxic_0 = toxic[toxic['toxic'] == 0].iloc[0:5000,:]

#concatenate toxic with non-toxic
toxic_balanced = pd.concat([toxic_1, toxic_0])

toxic_balanced['toxic'].value_counts()

"""STEP #5: MODEL TRAINING"""

def train_test(dataframe, label, vectorizer, ngram):
  X = dataframe.comment_text
  y = dataframe[label]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

  #removing stopwords and vectorizing
  rem_stp = vectorizer(ngram_range=(ngram), stop_words='english')

  X_train_rem_stp = rem_stp.fit_transform(X_train)
  X_test_rem_stp = rem_stp.transform(X_test)

  #Logistic Regression Alg
  log_r = LogisticRegression()
  log_r.fit(X_train_rem_stp, y_train)

  # #KNN Alg
  # knn = KNeighborsClassifier(n_neighbors=5)
  # knn.fit(X_train_rem_stp, y_train)

  f1_score_data = {'F1 Score':[f1_score(log_r.predict(X_test_rem_stp), y_test)]}

  f1_dataframe = pd.DataFrame(f1_score_data, index=["Logistic Regression"])

  return f1_dataframe

"""STEP #6: MODEL EVALUATION"""

insult_cv = train_test(insult_balanced, 'insult', TfidfVectorizer, (1,1))
insult_cv.rename(columns={"F1 Score": "F1 Score(insult)"}, inplace=True)
# insult_cv

identity_hate_cv = train_test(identity_hate_balanced, 'identity_hate', TfidfVectorizer, (1,1))
identity_hate_cv.rename(columns={"F1 Score": "F1 Score(identity_hate)"}, inplace=True)
# identity_hate_cv

obscene_cv = train_test(obscene_balanced, 'obscene', TfidfVectorizer, (1,1))
obscene_cv.rename(columns={"F1 Score": "F1 Score(identity_hate)"}, inplace=True)
# obscene_cv

threat_cv = train_test(threat_balanced, 'threat', TfidfVectorizer, (1,1))
threat_cv.rename(columns={"F1 Score": "F1 Score(threat)"}, inplace=True)
# threat_cv

severe_toxic_cv = train_test(severe_toxic_balanced, 'severe_toxic', TfidfVectorizer, (1,1))
severe_toxic_cv.rename(columns={"F1 Score": "F1 Score(severe_toxic)"}, inplace=True)
# severe_toxic_cv

toxic_cv = train_test(toxic_balanced, 'toxic', TfidfVectorizer, (1,1))
toxic_cv.rename(columns={"F1 Score": "F1 Score(toxic)"}, inplace=True)
# toxic_cv





textComment = ["He is very big and stupid"]


'''
Insult Probability
'''
X = insult_balanced.comment_text
y = insult_balanced['insult']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec2 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec2.fit_transform(X_train)
X_test_fit = tfVec2.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

commentVector2 = tfVec2.transform(textComment)

#insult
# insult_test1_vec = tfVec2.transform(general_comment)
print(f"Insult = {logistic_reg.predict_proba(commentVector2)[:,1]}")



'''
Identity Hate Probability
'''
X = identity_hate_balanced.comment_text
y = identity_hate_balanced['identity_hate']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec6 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec6.fit_transform(X_train)
X_test_fit = tfVec6.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

#identity hate
commentVector6 = tfVec6.transform(textComment)
print(f"Identity Hate = {logistic_reg.predict_proba(commentVector6)[:,1]}")




'''
Obscene Probability
'''
X = obscene_balanced.comment_text
y = obscene_balanced['obscene']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec4 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec4.fit_transform(X_train)
X_test_fit = tfVec4.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

#obscene
commentVector4 = tfVec4.transform(textComment)
print(f"Obscenity = {logistic_reg.predict_proba(commentVector4)[:,1]}")


'''
Threat Probability
'''
X = threat_balanced.comment_text
y = threat_balanced['threat']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec5 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec5.fit_transform(X_train)
X_test_fit = tfVec5.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

#threat
commentVector5 = tfVec5.transform(textComment)
print(f"Threat = {logistic_reg.predict_proba(commentVector5)[:,1]}")




'''
Severe Toxic Probability
'''
X = severe_toxic_balanced.comment_text
y = severe_toxic_balanced['severe_toxic']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec3 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec3.fit_transform(X_train)
X_test_fit = tfVec3.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

#severe toxic
commentVector3 = tfVec3.transform(textComment)
print(f"Severe Toxicity = {logistic_reg.predict_proba(commentVector3)[:,1]}")




'''
Toxic Probability
'''
X = toxic_balanced.comment_text
y = toxic_balanced['toxic']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=50)

#vectorizer
tfVec1 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfVec1.fit_transform(X_train)
X_test_fit = tfVec1.transform(X_test)
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train_fit, y_train)
logistic_reg.predict(X_test_fit)

commentVector1 = tfVec1.transform(textComment)
print(f"Toxicity = {logistic_reg.predict_proba(commentVector1)[:,1]}")




pickle.dump(logistic_reg, open('logistic_reg.pkl', 'wb'))
