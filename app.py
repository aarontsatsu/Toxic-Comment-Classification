from flask import Flask,request, render_template
import pickle
import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

training_set = pd.read_csv(
  "C:/Users/HP/OneDrive - Ashesi University/L200/SECOND SEMESTER/Intro to AI/Group 4 Project/Toxic Comment Classifier/Notebook/toxic_comment_dataset.csv"
  )


#using regex to remove irrelelvant characters
rem_alph_num = lambda x: re.sub('\w*\d\w*', ' ', x)
text_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
rem_newline = lambda x: re.sub("\n", ' ', x)
rem_nonascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
training_set['comment_text'] = training_set['comment_text'].map(rem_alph_num).map(text_lower).map(rem_newline).map(rem_nonascii)

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



#select 7877 insult comments
insult_1 = insult[insult['insult'] == 1].iloc[0:7877,:]

#select 7877 non insult comments
insult_0 = insult[insult['insult'] == 0].iloc[0:7877,:]

insult_balanced = pd.concat([insult_1, insult_0])



identity_hate_1 = identity_hate[identity_hate['identity_hate'] == 1].iloc[0:1405,:]

#select 1405 identity hate comments
identity_hate_0 = identity_hate[identity_hate['identity_hate'] == 0].iloc[0:1405,:]

#concatenate identity hate with non-identity hate
identity_hate_balanced = pd.concat([identity_hate_1, identity_hate_0])




#select 8449 obscene comments
obscene_1 = obscene[obscene['obscene'] == 1].iloc[0:8449,:]

#select 8449 non obscene comments
obscene_0 = obscene[obscene['obscene'] == 0].iloc[0:8449,:]

#concatenate obscene with non-obscene
obscene_balanced = pd.concat([obscene_1, obscene_0])




#select 478 threat comments
threat_1 = threat[threat['threat'] == 1].iloc[0:478,:]

#select 478 threat comments
threat_0 = threat[threat['threat'] == 0].iloc[0:478,:]

#concatenate identity hate with non-identity hate
threat_balanced = pd.concat([threat_1, threat_0])




#select 1595 severe toxic comments
severe_toxic_1 = severe_toxic[severe_toxic['severe_toxic'] == 1].iloc[0:1595,:]

#select 500 non severe toxic comments
severe_toxic_0 = severe_toxic[severe_toxic['severe_toxic'] == 0].iloc[0:1595,:]

#concatenate severe toxic with non-severe toxic
severe_toxic_balanced = pd.concat([severe_toxic_1, severe_toxic_0])




#select 5000 toxic comments
toxic_1 = toxic[toxic['toxic'] == 1].iloc[0:5000,:]

#select 500 non toxic comments
toxic_0 = toxic[toxic['toxic'] == 0].iloc[0:5000,:]

#concatenate toxic with non-toxic
toxic_balanced = pd.concat([toxic_1, toxic_0])





app = Flask(__name__)

model=pickle.load(open('logistic_reg.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("Classifier Interface.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    X1 = insult_balanced.comment_text
    y1 = insult_balanced['insult']

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.3, random_state=50)

    #vectorizer
    tfVec1 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec1.fit_transform(X_train1)
    X_test_fit = tfVec1.transform(X_test1)

    features1=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    input_vector = tfVec1.transform(features1)



    X2 = identity_hate_balanced.comment_text
    y2 = identity_hate_balanced['identity_hate']

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size=0.3, random_state=50)

    #vectorizer
    tfVec2 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec2.fit_transform(X_train2)
    X_test_fit = tfVec2.transform(X_test2)

    features2=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    input_vector1 = tfVec2.transform(features2)




    X3 = obscene_balanced.comment_text
    y3 = obscene_balanced['obscene']

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3,y3,test_size=0.3, random_state=50)

    #vectorizer
    tfVec3 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec3.fit_transform(X_train3)
    X_test_fit = tfVec3.transform(X_test3)

    features3=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    input_vector2 = tfVec3.transform(features3)



    
    X4 = threat_balanced.comment_text
    y4 = threat_balanced['threat']

    X_train4, X_test4, y_train4, y_test4 = train_test_split(X4,y4,test_size=0.3, random_state=50)

    #vectorizer
    tfVec4 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec4.fit_transform(X_train4)
    X_test_fit = tfVec4.transform(X_test4)

    features4=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    input_vector3 = tfVec4.transform(features4)




    X5 = severe_toxic_balanced.comment_text
    y5 = severe_toxic_balanced['severe_toxic']

    X_train5, X_test5, y_train5, y_test5 = train_test_split(X5,y5,test_size=0.3, random_state=50)

    #vectorizer
    tfVec5 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec5.fit_transform(X_train5)
    X_test_fit = tfVec5.transform(X_test5)

    features5=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    input_vector4 = tfVec5.transform(features5)




    X6 = toxic_balanced.comment_text
    y6 = toxic_balanced['toxic']

    X_train6, X_test6, y_train6, y_test6 = train_test_split(X6,y6,test_size=0.3, random_state=50)

    #vectorizer
    tfVec6 = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

    X_train_fit = tfVec6.fit_transform(X_train6)
    X_test_fit = tfVec6.transform(X_test6)

    features6=[str(x) for x in request.form.values()]
    # Transform the input text into a vector of word counts
    print(features6)
    input_vector5 = tfVec6.transform(features6)


    # variables = {
    #     'pred1':model.predict(input_vector)[0],
    #     'pred2':model.predict(input_vector1)[0],
    #     'pred3':model.predict(input_vector2)[0],
    #     'pred4':model.predict(input_vector3)[0],
    #     'pred5':model.predict(input_vector4)[0], 
    #     'pred6':model.predict(input_vector5)[0]
    # }


    return render_template("Classifier Interface.html", pred6=model.predict(input_vector5)[0])


    # pred1=model.predict(input_vector)[0], pred2=model.predict(input_vector1)[0],pred3=model.predict(input_vector2)[0],pred4=model.predict(input_vector3)[0],pred5=model.predict(input_vector4)[0], 




if __name__ == '__main__':
    app.run(debug=True)
