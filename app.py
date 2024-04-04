import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import sklearn
# from pathlib import Path


ps=PorterStemmer()

# variables to fetch images
# current_dir=Path(__file__).parent if "__file__" in locals() else Path.cwd()

def transform_text(text):
    text=text.lower() #'hi how are you'
    text=nltk.word_tokenize(text) #['hi', 'how', 'are', 'you']

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]#cloning of list
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)



tfidf= pickle.load(open("vectorizer3.pkl",'rb'))

model=pickle.load(open("model3.pkl","rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if not input_sms.strip():  # Check if the input message is empty
        st.warning("Please enter a message.")

    else:
        # 1 preprocess
        transformed_sms = transform_text(input_sms)

        # 2 vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3 predict
        result = model.predict(vector_input)[0]


        # 4 Display
        if result==1:
            st.error('SPAM MESSAGE!', icon="🚨")
        else:
            st.success('NOT A SPAM MESSAGE', icon="✅")
