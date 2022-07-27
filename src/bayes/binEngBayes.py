import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Spitting the date into train and validation
from sklearn.model_selection import train_test_split

#DATA PREPROCESSING
import nltk
# Uncomment to download "stopwords"
#nltk.download("stopwords")
from nltk.corpus import stopwords

#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

##BAYESIAN CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

#Evaluation on validation set
from sklearn.metrics import accuracy_score, roc_curve, auc

#Data paths
translated_data = 'data/EXIST2021_translatedTraining.csv'
translated_test_data = 'data/EXIST2021_translatedTest.csv'

# Load data and set labels
data = pd.read_csv(translated_data)

#convert labels to integers
data['LabelTask1'] = data['task1'].apply(lambda x : 1 if x == 'sexist' else 0)

#English version
data = data[['id', 'English', 'LabelTask1']]
data = data.rename(columns= {'id':'id', 'English':'tweet', 'LabelTask1':'label'})

#Spanish version
#data = data.reindex(columns=['id', 'Spanish', 'LabelTask1'])
#data = data.rename(columns= {'id':'id', 'Spanish':'tweet', 'LabelTask1':'label'})

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
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(
        model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)

    return auc.mean()


res = pd.Series([get_auc_CV(MultinomialNB(alpha=i))
                 for i in np.arange(1, 10, 0.1)],
                index=np.arange(1, 10, 0.1))

best_alpha = np.round(res.idxmax(), 2)
print('Best alpha: ', best_alpha)

plt.plot(res)
plt.title('AUC vs. Alpha')
plt.xlabel('Alpha')
plt.ylabel('AUC')
plt.show()


#Evaluation on validation set
def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Compute predicted probabilities
nb_model = MultinomialNB(alpha=1.8)
nb_model.fit(X_train_tfidf, y_train)
probs = nb_model.predict_proba(X_val_tfidf)

# Evaluate the classifier
evaluate_roc(probs, y_val)