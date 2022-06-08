#This was the preprocessing needed for my second approach to bert
import pandas as pd
from sklearn.model_selection import train_test_split

translated_data = 'data/EXIST2021_translatedTraining.csv'
translated_test_data = 'data/EXIST2021_translatedTest.csv'
destination_folder = 'data/task1Eng'

train_test_ratio = 0.10
train_valid_ratio = 0.80

# Read raw data
df_translated = pd.read_csv(translated_data)

# Prepare columns

df_translated['LabelTask1'] = df_translated['task1'].apply(lambda x : 1 if x == 'sexist' else 0)
df_translated = df_translated.reindex(columns=['id', 'English', 'LabelTask1'])

#Spanish version
#df_translated = df_translated.reindex(columns=['id', 'Spanish', 'LabelTask1'])

# Split according to label
df_sexist = df_translated[df_translated['LabelTask1'] == 1]
df_nonsexist = df_translated[df_translated['LabelTask1'] == 0]

# Train-test split
df_sexist_full_train, df_sexist_test = train_test_split(df_sexist, train_size = train_test_ratio, random_state = 1)
df_nonsexist_full_train, df_nonsexist_test = train_test_split(df_nonsexist, train_size = train_test_ratio, random_state = 1)

# Train-valid split
df_sexist_train, df_sexist_valid = train_test_split(df_sexist_full_train, train_size = train_valid_ratio, random_state = 1)
df_nonsexist_train, df_nonsexist_valid = train_test_split(df_nonsexist_full_train, train_size = train_valid_ratio, random_state = 1)

# Concatenate splits of different labels
df_train = pd.concat([df_sexist_train, df_nonsexist_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_sexist_valid, df_nonsexist_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_sexist_test, df_nonsexist_test], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv(destination_folder + '/train.csv', index=False)
df_valid.to_csv(destination_folder + '/valid.csv', index=False)
df_test.to_csv(destination_folder + '/test.csv', index=False)