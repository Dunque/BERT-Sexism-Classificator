import fasttext
import re

import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#Data paths
translated_data = '../../data/EXIST2021_translatedTraining.csv'
translated_test_data = '../../data/EXIST2021_translatedTest.csv'
modelPath = "../../models/fastText/"
trainingPath = "../../data/fasttext/"

# Load data and set labels
data = pd.read_csv(translated_data)

#Spanish version
data = data[['id', 'Spanish', 'task2']]
data = data.rename(columns={'id': 'id', 'Spanish': 'tweet', 'task2': 'label'})


def remove_clutterES(text):
    # keep only words
    remove_links = re.sub(r"(https?\://)\S+", " link ", text)
    remove_tags = re.sub(r"(?:\@)\S+", " tag ", text)
    letters_only_text = re.sub("[^abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZáéíóúüç]", " ", remove_links)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("spanish"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    return " ".join(meaningful_words)


# Adding __label__ to each label
def add_label(label):
    return " __label__" + label


# Variable used for the final report
labels = data.label.values

# Extract text and labels from the dataframe, and preprocess them
text = [remove_clutterES(text) for text in data.tweet.values]
appended_labels = [add_label(label) for label in labels]

# Combining both sets to produce the dataset
final_dataset = [''.join(item) for item in zip(text, appended_labels)]

# Spitting the date into train and validation
X_train, X_val, Y_train, Y_val = train_test_split(final_dataset, labels, test_size=0.1, random_state=2020)

# Write to a file that fasText understands
trainFile = open(trainingPath + "fast.train", 'w', encoding="utf-8")
validFile = open(trainingPath + "fast.valid", 'w', encoding="utf-8")

for line in X_train:
    trainFile.write(line + "\n")

for line in X_val:
    validFile.write(line + "\n")

model = fasttext.train_supervised(input=trainingPath + "fast.train", autotuneValidationFile=trainingPath + "fast.valid")

predictions = []
for sentence in X_val:
    predictions.append(model.predict(sentence)[0][0].replace('__label__',''))

clas_rep_file = open((modelPath + "fastENGbin_report.txt"), "w")
clas_rep_file.write(classification_report(Y_val, predictions, digits=4))
