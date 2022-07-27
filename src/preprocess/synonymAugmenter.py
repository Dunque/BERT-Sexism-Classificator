### Load dependences
import pandas as pd
import nlpaug.augmenter.word as naw

# Script meant to be used after translating

#### Data Path
PathDataSet = "data/"
FileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + FileName + '.csv', sep=',', index_col=0)

#### See the Data
print(df_train.head())

print(df_train.groupby(['task2']).size())

# Get all tweets from all sexist categories
tweets = df_train.loc[df_train['task2'] != "non-sexist"]

# Synonym based text augmenter
aug_eng = naw.SynonymAug(aug_src='wordnet')
aug_esp = naw.SynonymAug(aug_src='wordnet', lang='spa')

for index, tweet in tweets.iterrows():
    df_train.loc[len(df_train)] = ['EXIST2021', len(df_train) + 1, tweet['source'], tweet['language'],
                                   tweet['text'], tweet['task1'], tweet['task2'],
                                   aug_eng.augment(tweet["English"]), aug_esp.augment(tweet["Spanish"])]


#### Save Dataset
PathDataSet = "data/"
NewFileName = "EXIST2021_translatedTrainingAugmented"
df_train.to_csv(PathDataSet + NewFileName + '.csv')

print(df_train.groupby(['task2']).size())