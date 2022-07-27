# pip install pandas
# pip install easynmt

### Load dependences
import pandas as pd
from easynmt import EasyNMT

# Translation model
translator = EasyNMT('opus-mt', max_loaded_models=10)


### TRAINING DATASET

#### Data Path
PathDataSet = "data/"
FileName = "EXIST2021_training.tsv"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + FileName, sep='\t')

#### See the Data
print(df_train.head())

### Create Columns for all comments in English after translation
df_train['English'] = ''
print(df_train.head())

### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 6976, 50):
  df_train.loc[index:index+51, ['English']] = df_train.loc[index:index+51].apply(lambda x: translator.translate(
      x['text'], source_lang='es', target_lang='en') if x['language'] == 'es' else x['text'], axis=1)

#### Save Dataset with 'English' column
PathDataSet = "data/"
NewFileName = 'EXIST2021_translatedTraining'
df_train.to_csv(PathDataSet + NewFileName + '.csv')


#### Read/Load Dataset
#### Data Path
PathDataSet = "data/"
NewFileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

print(df_train.head())

### Create Columns for all comments in Spanish after translation
df_train['Spanish'] = ''
print(df_train.head())

### Fill the columns 'Spanish' with English comments translated to Spanish and original Spanish comments
for index in range(0, 6976, 50):
  df_train.loc[index:index+51, ['Spanish']] = df_train.loc[index:index+51].apply(lambda x: translator.translate(
      x['text'], source_lang='en', target_lang='es') if x['language'] == 'en' else x['text'], axis=1)

#### Save Dataset with 'Spanish' column
PathDataSet = "data/"
NewFileName = 'EXIST2021_translatedTraining'
df_train.to_csv(PathDataSet + NewFileName + '.csv')

#### Read/Load Dataset
#### Data Path
PathDataSet = "data/"
NewFileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

print(df_train.head())


### TEST DATASET


#### Data Path
PathDataSet = "data/"
FileName = "EXIST2021_test.tsv"
#### Load tsv as a Data Frame
df_test = pd.read_csv(PathDataSet + FileName, sep='\t')

#### See the Data
print(df_test.head())


### Create Columns for all comments in English after translation
df_test['English'] = ''
print(df_test.head())

### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 4367, 100):
  df_test.loc[index:index+101, ['English']] = df_test.loc[index:index+101].apply(
    lambda x: translator.translate(x['text'], source_lang='es', target_lang='en') if x['language'] == 'es' else x['text'], axis=1)

#### Save Dataset with 'English' column
PathDataSet = "data/"
NewFileName = 'EXIST2021_translatedTest'
df_test.to_csv(PathDataSet + NewFileName + '.csv')

#### Read/Load Dataset
#### Data Path
PathDataSet = "data/"
NewFileName = "EXIST2021_translatedTest"
#### Load tsv as a Data Frame
df_test = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

print(df_test.tail())

df_test['Spanish'] = ''
print(df_test.head())

### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 4367, 100):
  df_test.loc[index:index+101, ['Spanish']] = df_test.loc[index:index+101].apply(
    lambda x: translator.translate(x['text'], source_lang='en', target_lang='es') if x['language'] == 'en' else x['text'], axis=1)

#### Save Dataset with 'Spanish' column
PathDataSet = "data/"
NewFileName = "EXIST2021_translatedTest"
df_test.to_csv(PathDataSet + NewFileName + '.csv')

print(df_test.head())
