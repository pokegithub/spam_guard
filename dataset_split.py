import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# stratify makes sure both splits have the same spam/ham ratio
df_train, df_test = train_test_split(df, test_size=0.3, random_state=36, stratify=df['label'])

print("Training set size:", df_train.shape)
print("Testing set size:", df_test.shape)
print("\nTraining label counts:\n", df_train['label'].value_counts())
print("\nTesting label counts:\n", df_test['label'].value_counts())

df_train.to_csv('training_data.csv', index=False)
df_test.to_csv('testing_data.csv', index=False)
print("\nSaved training_data.csv and testing_data.csv")
