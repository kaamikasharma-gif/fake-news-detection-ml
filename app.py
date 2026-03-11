import joblib
import streamlit as st

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📰 Fake News Detection System")

st.write("Paste a news article below and the model will predict if it is **Fake or Real**.")

user_input = st.text_area("Enter News Article")

if st.button("Analyze News"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        if prediction == 0:
            st.error("Prediction: Fake News ")
        else:
            st.success("Prediction: Real News ")
