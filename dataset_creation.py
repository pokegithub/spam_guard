import pandas as pd

df1 = pd.read_csv('datasets/email.csv')
df2 = pd.read_csv('datasets/emails.csv')
df3 = pd.read_csv('datasets/combined_data.csv')

"""print("df1 details:")
print(df1.shape)
print(df1.info())
print(df1.columns)
print()
print("df2 details:")
print(df2.shape)
print(df2.info())
print(df2.columns)
print()
print("df3 details:")
print(df3.shape)
print(df3.info())
print(df3.columns)"""

df1 = df1.rename(columns = {'Category': 'label', 'Message': 'text'})
df2 = df2.rename(columns = {'spam': 'label'})
for i in range(2, 110):
    del df2[f'Unnamed: {i}']

"""print("df1 columns:")
print(df1.columns)
print()
print("df2 columns:")
print(df2.columns)
print()
print("df3 columns:")
print(df3.columns)"""

df1 = df1[['text', 'label']]
df3 = df3[['text', 'label']]

"""print("df1 columns:")
print(df1.columns)
print()
print("df2 columns:")
print(df2.columns)
print()
print("df3 columns:")
print(df3.columns)"""

combined_df = pd.concat([df1, df2, df3], ignore_index=True)

print(combined_df.shape)
print(combined_df.info())
print(combined_df.columns)

combined_df.to_csv("data.csv", index = False)