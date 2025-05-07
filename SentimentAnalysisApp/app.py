import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

#Load model
with open("sen_ana_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
with open("sen_ana_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Phân tích bình luận người dùng")

user_comment = st.text_area("Nhập bình luận...", "")
user_comment = user_comment.strip()

if st.button("Phân tích"):
    if user_comment == "":
        st.warning("Xin nhập bình luận")
    else:
        #Vectorize the input
        input_vector = vectorizer.transform([user_comment])
        #Predict sentiment
        prediction = model.predict(input_vector)[0]
        #Display the result
        if prediction == 1:
            sentiment_label = "Tích cực"
        elif prediction == -1:
            sentiment_label = "Tiêu cực"
        else:
            sentiment_label = "Trung lập"
        st.success(f"Dự đoán cảm xúc: **{sentiment_label}**")
