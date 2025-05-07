import nltk
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import contractions
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import streamlit as st
import pickle
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Buoc 2: Load dataset va xu ly du lieu
# lưu trữ
# đọc file train
try:
    train_df = pd.read_csv('train_dataset_en_comments.csv', encoding='utf-8')
    print("Đọc thành công bằng encoding utf-8")
except UnicodeDecodeError:
    try:
        train_df = pd.read_csv('train_dataset_en_comments.csv', encoding='ISO-8859-1')
        print("Đọc thành công bằng encoding ISO-8859-1")
    except UnicodeDecodeError:
        train_df = pd.read_csv('train_dataset_en_comments.csv', encoding='cp1252')
        print("Đọc thành công bằng encoding cp1252")

# lưu trữ
# đọc file test
try:
    test_df = pd.read_csv('test_dataset_en_comments.csv', encoding='utf-8')
    print("Đọc thành công bằng encoding utf-8")
except UnicodeDecodeError:
    try:
        test_df = pd.read_csv('test_dataset_en_comments.csv', encoding='ISO-8859-1')
        print("Đọc thành công bằng encoding ISO-8859-1")
    except UnicodeDecodeError:
        test_df = pd.read_csv('test_dataset_en_comments.csv', encoding='cp1252')
        print("Đọc thành công bằng encoding cp1252")

# Bước 3: tạo labels mới và shuffle data
train_df = shuffle(train_df).reset_index(drop=True)
test_df = shuffle(test_df).reset_index(drop=True)


# one hot => only one label
def convert_label(row):
    if row['positive'] == 1:
        return 'positive'
    elif row['neutral'] == 1:
        return 'neutral'
    elif row['negative'] == 1:
        return 'negative'
    else:
        return 'unknown'


train_df['label'] = train_df.apply(convert_label, axis=1)
test_df['label'] = test_df.apply(convert_label, axis=1)

train_df_reduced = train_df[['comment', 'label']].copy()
test_df_reduced = test_df[['comment', 'label']].copy()

sample_text = train_df.loc[1, 'comment']
test_text = "brb, don't forget doesn't important bpd med isn't cool aren't right"


# Buoc 4: Chuyen ve chu thuong
def to_lower(text):
    return text.lower()


lower_text = to_lower(sample_text)
print("lower_ test_text: ", to_lower(test_text))
print("lowered sample: ", lower_text)


# Buoc 5: Mo rong tu viet ngan
def expand_contractions(text):  # mo rong cac tu viet tat nhu isn't
    return contractions.fix(text)


contraction_expanded_text = expand_contractions(lower_text)
print("Mo rong cac tu viet tat cua mau sample: ", contraction_expanded_text)
print("Mo rong cac tu viet tat cua mau sample: ", expand_contractions(test_text))

# mo rong chu viet ngan/teencode
textFile = open('teencode.txt', encoding='UTF-8').read()  # doc file
# chuyen kieu string thanh dict
textDict = ast.literal_eval(textFile)


def expand_teencode(text):  # mo rong may tu teencode cua gioi tre nhu btw
    words = text.split()
    # tam thoi xoa dau cau de xu ly teencode
    expandedWords = []
    for word in words:
        wordWithoutPunctuation = word.strip(string.punctuation)
        expandedTeencode = textDict.get(wordWithoutPunctuation, wordWithoutPunctuation)
        # tra lai dau cau
        if word[-1] in string.punctuation:
            expandedTeencode += word[-1]
        expandedWords.append(expandedTeencode)
    return ' '.join(expandedWords)


teencodeExpandedSample = expand_teencode(contraction_expanded_text)
print("Mo rong cac tu teen code: ", teencodeExpandedSample)
print("Mo rong cac tu teen code: ", expand_teencode(expand_contractions(test_text)))


# Buoc 6: Tokenizing
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


token_sample = tokenize_text(teencodeExpandedSample)
print("Cac tu trong cau sample_text: ", token_sample)
print("Cac tu trong cau test_text: ", tokenize_text(expand_teencode(expand_contractions(test_text))))


# Buoc 7: Bo dau cau va tu ko phai chu
def remove_punctuation(tokens):
    # isalpha() chi giu lai a-z, A-Z, con lai xoa sach. xoa luon ca dau truoc chu (i'm -> i)
    cleaned_tokens = [word for word in tokens if word.isalpha()]
    return cleaned_tokens


clean_token_sample = remove_punctuation(token_sample)
print("Sau khi bo dau cau va cac tu ko phai chu trong sample_text: ", clean_token_sample)
print("Sau khi bo dau cau va cac tu ko phai chu trong test_text: ",
      remove_punctuation(tokenize_text(expand_teencode(expand_contractions(test_text)))))

# Buoc 8: Bo stopwords:
stop_words = set(stopwords.words('english'))
negation_words = {"no", "nor", "not", "don", "don't", "ain", "aren", "aren't",
                  "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
                  "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
                  "isn", "isn't", "mightn", "mightn't", "mustn", "mustn't",
                  "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't",
                  "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
                  }

# loai bo cac tu mang y phu dinh trong thu vien stop_words
stop_words = stop_words - negation_words


# loai bo stopwords (giu lai tu mang nghia phu dinh)
def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


filtered_token_sample = remove_stopwords(clean_token_sample)
print("Sau khi bo stopwords trong sample_text: ", filtered_token_sample)
print("Sau khi bo stopwords trong test_text: ",
      remove_stopwords(remove_punctuation(tokenize_text(expand_teencode(expand_contractions(test_text))))))

# Buoc 9: Lemmatization:
lemmatizer = WordNetLemmatizer()


def lemmatize_text(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens


lemmatize_token_sample = lemmatize_text(filtered_token_sample)
print("Sau khi lemmatize: ", lemmatize_token_sample)

final_text = " ".join(lemmatize_token_sample)
print("Sample text sau khi hoàn tat xu ly: ", final_text)
print("Test text sau khi hoàn tat xu ly: ", " ".join(lemmatize_text(
    remove_stopwords(remove_punctuation(tokenize_text(expand_teencode(expand_contractions(test_text))))))))

print("========================================================================")


# Ap dung len dataframe cac ham xu ly
def preprocess_text(text):
    text = to_lower(text)
    text = expand_contractions(text)
    text = expand_teencode(text)
    tokens = tokenize_text(text)
    clean_tokens = remove_punctuation(tokens)
    filtered_tokens = remove_stopwords(clean_tokens)
    lemmatize_tokens = lemmatize_text(filtered_tokens)
    final_text = " ".join(lemmatize_tokens)
    return final_text


train_df_reduced['preprocessed_text'] = train_df_reduced['comment'].apply(preprocess_text)
print("Train dataset duoc xu ly: ", train_df_reduced[['comment', 'preprocessed_text']].head())

test_df_reduced['preprocessed_text'] = test_df_reduced['comment'].apply(preprocess_text)
print("Test dataset duoc xu ly: ", test_df_reduced[['comment', 'preprocessed_text']].head())

# Extract features
# khởi tạo CountVectorizer
vectorizer = CountVectorizer()
# apply len processed comment
vectorized_train_text = vectorizer.fit_transform(train_df_reduced['preprocessed_text'])
vectorized_test_text = vectorizer.transform(test_df_reduced['preprocessed_text'])
# dua ve thanh dataframe de co the quan sat ket qua
vectorized_train_text_df = pd.DataFrame(vectorized_train_text.toarray(), columns=vectorizer.get_feature_names_out())
print(vectorized_train_text_df.head(5))
vectorized_test_text_df = pd.DataFrame(vectorized_test_text.toarray(), columns=vectorizer.get_feature_names_out())
print(vectorized_test_text_df.head(5))

# Build label for model
# map label
label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
# chuyen label ve dang so
numberized_train_label = train_df_reduced['label'].map(label_mapping)
numberized_test_label = test_df_reduced['label'].map(label_mapping)

print(numberized_train_label[:5])
print(numberized_test_label[:5])

# Build Model (NaiveBayes)

# tach features rieng cho test va train
x_train = vectorized_train_text_df
y_train = numberized_train_label

x_test = vectorized_test_text_df
y_test = numberized_test_label
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# huan luyen model
nb_model = MultinomialNB()  # khoi tao mo hinh Naive Bayes

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb_model', MultinomialNB())
])

nb_model.fit(x_train, y_train)  # huan luyen mo hinh
y_pred = nb_model.predict(x_test)

# kết quả và đánh giá
print(f"Độ chính xác mô hình: {accuracy_score(y_test, y_pred) * 100:.4f}%")
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))
print("Ma trận nhầm lẫn")
print(confusion_matrix(y_test, y_pred))

# Dữ liệu thực tế muốn kiểm tra
sample_test = "I hate everyone"

# Tiền xử lý dữ liệu (giống như đã làm với tập huấn luyện)
sample_test_preprocessed = preprocess_text(sample_test)
print(sample_test_preprocessed)

# Chuyển dữ liệu mẫu thành vectơ sử dụng CountVectorizer đã huấn luyện trước đó
sample_test_vectorized = vectorizer.transform([sample_test_preprocessed])

# Dự đoán nhãn của mẫu dữ liệu thực tế
predicted_label = nb_model.predict(sample_test_vectorized)

# Chuyển nhãn số thành nhãn văn bản (ví dụ: positive, neutral, negative)
label_mapping_reverse = {1: 'positive', 0: 'neutral', -1: 'negative'}
predicted_label_text = label_mapping_reverse[predicted_label[0]]

print("Predicted Label (Text):", predicted_label_text)

#Truc quan hoa du lieu
plt.figure(figsize=(8,6))
sns.countplot(x='label', data=train_df_reduced)
plt.title("Distribution of Labels in Training Data")
plt.savefig('graph1.png')
plt.show()

# dataset test
plt.figure(figsize=(8,6))
sns.countplot(x='label', data=test_df_reduced)
plt.title("Distribution of Labels in Test Data")
plt.savefig('graph2.png')
plt.show()

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Trực quan hóa ma trận nhầm lẫn
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig('graph3.png')
plt.show()

#Lay danh sach cac tu tu vectorizer
words = np.array(vectorizer.get_feature_names_out())

#Lay so lan xuat hien cua cac tu trong tap huan luyen
word_counts = np.array(vectorized_train_text.sum(axis=0)).flatten()

#Tao dataframe tu ket qua
word_freq_df = pd.DataFrame({'Word': words, 'Count': word_counts})
word_freq_df = word_freq_df.sort_values(by='Count', ascending=False)

#Truc quan hoa 20 tu xuat hien nhieu nhat
plt.figure(figsize=(10,6))
sns.barplot(x='Count', y='Word', data=word_freq_df.head(20))
plt.title("Top 20 Most Frequent Words in Training Data")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.savefig("graph4.png")
plt.show()


# Extract model
#Luu mo hinh da xuat hien
joblib.dump(nb_model, 'sen_ana_model.pkl')

#Luu vectorizer (neu co)
joblib.dump(vectorizer, 'sen_ana_vectorizer.pkl')

# Tải vectorizer (nếu có)
#files.download('sen_ana_vectorizer.pkl)
file_path_vectorizer = 'sen_ana_vectorizer.pkl'
try:
    with open("sen_ana_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
        print("Save vectorizer successfully")
except FileNotFoundError:
    print(f"File not found: {file_path_vectorizer}")
except pickle.UnpicklingError:
    print("Error: The file content is not a valid pickle format.")
except EOFError:
    print("Error: The file is incomplete or corrupted")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Tải mô hình
#files.download('sen_ana_model.pkl')
file_path_model = 'sen_ana_model.pkl'
try:
    with open("sen_ana_model.pkl", "wb") as file:
        pickle.dump(nb_model, file)
        print("Save model successfully")
except FileNotFoundError:
    print(f"File not found: {file_path_vectorizer}")
except pickle.UnpicklingError:
    print("Error: The file content is not a valid pickle format.")
except EOFError:
    print("Error: The file is incomplete or corrupted")
except Exception as e:
    print(f"An unexpected error occurred: {e}")