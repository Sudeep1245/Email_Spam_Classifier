import streamlit as st
import pickle
import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer






ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    word_list = nltk.word_tokenize(text,language='english', preserve_line=True)
    y=[]
    for i in word_list:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i  not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

with open('vectorizer.pkl', 'rb') as f:
    tdfidf = pickle.load(f)

with open('model.pkl','rb') as f1:
    model = pickle.load(f1)

st.title('Email/SMS SPAM CLASSIFIER')
input_sms = st.text_input('Enter The Message')

if st.button('Predict'):

    transformed_txt = transform_text(input_sms)
    vector_input = tdfidf.transform([transformed_txt])
    result = model.predict(vector_input)[0]
    if result ==1:
        st.header('Spam')
    else :
        st.header('Not Spam')