### Load dependences
import pandas as pd
import nlpaug.augmenter.word as naw

# Script meant to be used after translating

#### Data Path
PathDataSet = "../../data/"
FileName = "EXIST2021_translatedTraining"
#### Load tsv as a Data Frame
df_train = pd.read_csv(PathDataSet + FileName + '.csv', sep=',', index_col=0)

#### See the Data
print(df_train.head())

print(df_train.groupby(['task2']).size())

# Get all tweets from all sexist categories
tweets = df_train.loc[df_train['task2'] != "non-sexist"]

# Synonym based text augmenter
# aug_eng = naw.SynonymAug(aug_src='wordnet')
#
# aug_esp = naw.SynonymAug(aug_src='wordnet', lang='spa')
#
# i = 0
#
# for index, tweet in tweets.iterrows():
#     df_train.loc[len(df_train)] = ['EXIST2021', len(df_train) + 1, tweet['source'], tweet['language'],
#                                    tweet['text'], tweet['task1'], tweet['task2'],
#                                    aug_eng.augment(tweet["English"]), aug_esp.augment(tweet["Spanish"])]
#     i = i+1
#     print("done {}".format(i))
#
# print("finished synonyms")

 # Repeat the process but now using contextual word embeddings

print(len(tweets))
i = 0
# Synonym based text augmenter
back_translation_aug_eng = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-en-de',
    to_model_name='Helsinki-NLP/opus-mt-de-en')

back_translation_aug_esp = naw.BackTranslationAug(
    from_model_name='Helsinki-NLP/opus-mt-es-it',
    to_model_name='Helsinki-NLP/opus-mt-it-es')

aug_eng2 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

aug_esp2 = naw.ContextualWordEmbsAug(model_path='SpanBERT/spanbert-large-cased', action="substitute")

for index, tweet in tweets.iterrows():
    df_train.loc[len(df_train)] = ['EXIST2021', len(df_train) + 1, tweet['source'], tweet['language'],
                                   tweet['text'], tweet['task1'], tweet['task2'],
                                   back_translation_aug_eng.augment(tweet["English"]),
                                   back_translation_aug_esp.augment(tweet["Spanish"])]
    i = i+1
    print("done {}".format(i))


#### Save Dataset
NewFileName = "EXIST2021_translatedTrainingAugmented2"
df_train.to_csv(PathDataSet + NewFileName + '.csv')

print(df_train.groupby(['task2']).size())