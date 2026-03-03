import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import joblib

df = pd.read_csv('preprocessed_training_data.csv')
df1 = pd.read_csv('preprocessed_testing_data.csv')

x = df['clean_text']
x1 = df1['clean_text']
y = df['label']
y1 = df1['label']

tfidf = TfidfVectorizer(
    max_features=5000,  # keeping only top 5,000 most frequent/important words
    min_df=5,           # ignoring words that appear in <5 emails
    ngram_range=(1, 1)  # (1,1) -> single words
)

x_vec = tfidf.fit_transform(tqdm(x, desc = 'Training: TF-IDF Vectorization'))
x1_vec = tfidf.transform(tqdm(x1, desc = 'Testing: TF-IDF Vectorization'))

joblib.dump(x_vec, 'x_training_vector.pkl')
joblib.dump(x1_vec, 'x_testing_vector.pkl')
joblib.dump(y, 'y_training_vector.pkl')
joblib.dump(y1, 'y_testing_vector.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')


#print(x_vec[0])
