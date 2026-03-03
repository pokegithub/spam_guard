import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 

tqdm.pandas() 

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Custom stopwords to remove loose URL fragments
custom_stopwords = ['http', 'https', 'www', 'com', 'org', 'net', 'co']
stop_words.update(custom_stopwords)
lemmatizer = WordNetLemmatizer()

print("Loading dataset...")
df = pd.read_csv('training_data.csv')
df1 = pd.read_csv('testing_data.csv')

df['text'] = df['text'].str.replace('Subject: ', '')
df1['text'] = df1['text'].str.replace('Subject: ', '')

df['label'] = df['label'].replace({'0': 'ham', '1': 'spam'})
df = df[df['label'].isin(['ham', 'spam'])]
df['label'] = df['label'].replace({'ham': 0, 'spam': 1})

df1['label'] = df1['label'].replace({'0': 'ham', '1': 'spam'})
df1 = df1[df1['label'].isin(['ham', 'spam'])]
df1['label'] = df1['label'].replace({'ham': 0, 'spam': 1})

print("Dropping duplicates...")
df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
df1 = df1.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

print("Calculating character counts...")
df['num_chars'] = df['text'].progress_apply(len)
df1['num_chars'] = df1['text'].progress_apply(len)

print("Calculating word counts...")
df['num_words'] = df['text'].progress_apply(lambda x: len(nltk.word_tokenize(str(x))))
df1['num_words'] = df1['text'].progress_apply(lambda x: len(nltk.word_tokenize(str(x))))

print("Calculating sentence counts...")
df['num_sentences'] = df['text'].progress_apply(lambda x: len(nltk.sent_tokenize(str(x))))
df1['num_sentences'] = df1['text'].progress_apply(lambda x: len(nltk.sent_tokenize(str(x))))


def clean_text(text):
    # Added str() here just in case any completely blank cells slipped through
    # This prevents the float attribute error
    text = str(text).lower()

    url_regex = r'(https?:\/\/\S+|www\.\S+)'
    email_regex = r'[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}'
    num_regex = r'\d+'
    currency_regex = r'[$£€]'

    text = re.sub(url_regex, 'webaddr', text)
    text = re.sub(email_regex, 'emailaddr', text)
    text = re.sub(num_regex, 'num', text)
    text = re.sub(currency_regex, 'currency ', text)

    text = text.replace('escapenumber', 'num')

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    # len(i) > 1 to filter out leftover tags such as x, e, etc. and seen in EDA
    clean_words = [lemmatizer.lemmatize(i) for i in words if i not in stop_words and len(i) > 1]

    return ' '.join(clean_words)


print("Running text preprocessing...")
df['clean_text'] = df['text'].progress_apply(clean_text)
df1['clean_text'] = df1['text'].progress_apply(clean_text)

print("Checking for nulls in clean_text:")
print(df['clean_text'].isnull().sum())
print(df1['clean_text'].isnull().sum())

# Removing empty strings
df = df[df['clean_text'].str.strip() != '']
df1 = df1[df1['clean_text'].str.strip() != '']

df = df[['text', 'clean_text', 'label', 'num_chars', 'num_words', 'num_sentences']]
df1 = df1[['text', 'label', 'clean_text', 'num_chars', 'num_words', 'num_sentences']]

print(df.columns)
print(df1.columns)

print("Saving preprocessed data to CSV...")
df.to_csv('preprocessed_training_data.csv', index = False)
df1.to_csv('preprocessed_testing_data.csv', index = False)
# index = False: prevents new column to be added as index


print("Data preprocessing complete.")
