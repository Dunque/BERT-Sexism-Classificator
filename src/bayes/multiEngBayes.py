import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Spitting the date into train and validation
from sklearn.model_selection import train_test_split

#DATA PREPROCESSING
import nltk
# Uncomment to download "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

##BAYESIAN CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

#Evaluation on validation set
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family': 'serif'})
plt.style.use("seaborn-whitegrid")

#Data paths
translated_data = '../../data/EXIST2021_translatedTrainingAugmented.csv'
translated_test_data = '../../data/EXIST2021_translatedTest.csv'
modelPath = "../../models/bayes/multiEng/"

# Load data and set labels
data = pd.read_csv(translated_data)

category_list = list(data.task2.unique())
category_list.remove('non-sexist')
category_list.insert(0, 'non-sexist')
category_sexism = {category_list[index]: index for index in range(len(list(data.task2.unique())))}
data['LabelTask2'] = data['task2'].apply(lambda x: category_sexism[x])

# English version
data = data[['id', 'English', 'LabelTask2']]
data = data.rename(columns={'id': 'id', 'English': 'tweet', 'LabelTask2': 'label'})

# Display 5 random samples
data.sample(5)

# Spitting the date into train and validation
X = data.tweet.values
y = data.label.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=2020)

# Load test data
test_data = pd.read_csv(translated_test_data)

# Keep important columns
# English
test_data = test_data[['id', 'English']]
test_data = test_data.rename(columns= {'id':'id', 'English':'tweet'})

# Spanish
#test_data = test_data[['id', 'Spanish']]
#test_data = test_data.rename(columns= {'id':'id', 'Spanish':'tweet'})

# Display 5 samples from the test data
test_data.sample(5)

#DATA PREPROCESSING (TF-IDF and BAYES)
def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove links
    s = re.sub(r"http\S+", "", s)
    
    return s

# Preprocess text
X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])

# Calculate TF-IDF
tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         smooth_idf=False)
X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
X_val_tfidf = tf_idf.transform(X_val_preprocessed)

##BAYESIAN CLASSIFIER
def get_auc_CV(model):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=2020)

    # Get f1_weighted scores
    f1 = cross_val_score(
        model, X_train_tfidf, y_train, scoring="f1_weighted", cv=kf)

    return f1.mean()


res = pd.Series([get_auc_CV(MultinomialNB(alpha=i))
                 for i in np.arange(0, 1, 0.001)],
                index=np.arange(0, 1, 0.001))

best_alpha = np.round(res.idxmax(), 2)
print('Best alpha: ', best_alpha)

plt.plot(res)
plt.xlabel('Alpha')
plt.ylabel('F1-score')

plt.tight_layout()
plt.savefig(modelPath + "alphaAUC.png")


# Compute predicted probabilities
nb_model = MultinomialNB(alpha=best_alpha)
nb_model.fit(X_train_tfidf, y_train)
probs = nb_model.predict_proba(X_val_tfidf)

# Evaluate the classifier
clas_rep_file = open((modelPath + "classReportAug.txt"), "w")
clas_rep_file.write(classification_report(y_val, np.argmax(probs, axis=1), target_names=["NS","II", "O", "SV", "SD", "MNSV"], digits=4))