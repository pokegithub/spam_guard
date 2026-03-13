import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')

# Split the dataset 70/30
df_training, df_testing = train_test_split(df, test_size = 0.3, random_state = 36)

print("Training Label Distribution:")
print(df_training['label'].value_counts())

# Save the files WITHOUT dropping the 'label' column!
df_training.to_csv('training_data.csv', index = False)
df_testing.to_csv('testing_data.csv', index = False)
