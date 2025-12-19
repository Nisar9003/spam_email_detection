import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# download stopwords if not present
nltk.download('stopwords')

# load model & vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# Streamlit UI
st.title("Email Spam Detection")
st.write("Logistic Regression based Spam Classifier")


email_text = st.text_area("Enter email text here:")

if st.button("Predict"):
    cleaned_text = clean_text(email_text)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("SPAM")
    else:
        st.success("HAM")
