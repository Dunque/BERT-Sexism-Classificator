# pip install googletrans==4.0.0-rc1
# pip install mtranslate

# Mount Google Drive
from google.colab import drive  # import drive from google colab

ROOT = "/content/drive"     # default location for the drive
print(ROOT)                 # print content of ROOT (Optional)

drive.mount(ROOT)           # we mount the google drive at /content/drive

### Load dependences
from googletrans import Translator
from mtranslate import translate
import time
import pandas as pd

# translator = Translator()

#### Data Path
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
FileName = "EXIST2021_training.tsv"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + FileName, sep='\t')

#### See the Data
df_train.head()

### Create Columns for all comments in English after translation
df_train['English'] = ''
df_train.head()


### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 6976, 50):
  from googletrans import Translator
  translator = Translator()
  df_train.loc[index:index+51, ['English']] = df_train.loc[index:index+51].apply(lambda x: translator.translate(
      x['text'], src='es', dest='en').text if x['language'] == 'es' else x['text'], axis=1)
  time.sleep(1)


#### Save Dataset with 'English' column
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = 'EXIST2021_translatedTraining'
df_train.to_csv(PathDataSet + NewFileName + '.csv')


#### Read/Load Dataset
#### Data Path
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

df_train.head()

### Create Columns for all comments in English after translation
df_train['Spanish'] = ''
df_train.head()

### Fill the columns 'Spanish' with English comments translated to Spanish and original Spanish comments
for index in range(0, 6976, 50):
  from googletrans import Translator
  translator = Translator()
  df_train.loc[index:index+51, ['Spanish']] = df_train.loc[index:index+51].apply(lambda x: translator.translate(
      x['text'], src='en', dest='es').text if x['language'] == 'en' else x['text'], axis=1)
  time.sleep(1)

#### Save Dataset with 'English' column
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = 'EXIST2021_translatedTraining'
df_train.to_csv(PathDataSet + NewFileName + '.csv')

#### Read/Load Dataset
#### Data Path
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

df_train.head()

#### Data Path
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
FileName = "EXIST2021_test.tsv"
#### Load tsv as a Data Frame
df_test = pd.read_csv(PathDataSet + FileName, sep='\t')

#### See the Data
df_test.head()


### Create Columns for all comments in English after translation
df_test['English'] = ''
df_test.head()


### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 4367, 100):
  print(index)
  df_test.loc[index:index+101, ['English']] = df_test.loc[index:index+101].apply(
      lambda x: translate(x['text'], 'en') if x['language'] == 'es' else x['text'], axis=1)
  time.sleep(1)

#### Save Dataset with 'English' column
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = 'EXIST2021_translatedTest'
df_test.to_csv(PathDataSet + NewFileName + '.csv')

#### Read/Load Dataset
#### Data Path
PathDataSet = "../content/drive/MyDrive/Code/EXITS/Data/"
NewFileName = "EXIST2021_translatedTest"
#### Load tsv as a Data Frame
df_test = pd.read_csv(PathDataSet + NewFileName + '.csv', index_col=0)

df_test.tail()

### Fill the columns 'English' with Spanish comments translated to English and original English comments
for index in range(0, 4367, 100):
  print(index)
  df_test.loc[index:index+101, ['Spanish']] = df_test.loc[index:index+101].apply(
      lambda x: translate(x['text'], 'es') if x['language'] == 'en' else x['text'], axis=1)
  time.sleep(1)

df_test.head()
