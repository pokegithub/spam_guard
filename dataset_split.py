import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

#df_training = df.sample(frac = 0.7, random_state = 36)
#df_testing = df.drop(df_training.index)

df_training, df_testing = train_test_split(df, test_size = 0.3, random_state = 36)

print(df_training['label'].value_counts())

#df_testing = df_testing.drop(columns = ['label'])

df_training.to_csv('training_data.csv', index = False)

df_testing.to_csv('testing_data.csv', index = False)
