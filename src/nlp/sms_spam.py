# course: https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset/data

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# use counts for calculating the features
def get_x_from_count_vectorizer():
    count_vectorizer = CountVectorizer(decode_error='ignore')
    x = count_vectorizer.fit_transform(df['data'])
    return x


# use Tfidf vectorizer for calculating the features
def get_x_from_tfidf_vectorizer():
    tfidf_vectorizer = TfidfVectorizer(decode_error='ignore')
    x = tfidf_vectorizer.fit_transform(df['data'])
    return x


# train the input model with input data and print the scores
def train_and_print_scores(cur_model, x, y):
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    cur_model.fit(x_train, y_train)

    print("train scores:", cur_model.score(x_train, y_train))
    print("test scores:", cur_model.score(x_test, y_test))


df = pd.read_csv('data/sms-spam-collection/spam.csv', encoding='ISO-8859-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
# shuffling is not improving the quality of input and so commenting it out
# df = df.sample(frac=1).reset_index(drop=True)
Y = df['b_labels']

# use count vectorizer for calculating the features
print("CountVectorizer for features")
print("============================")
X = get_x_from_count_vectorizer()
# use naive bayes classifier
print("Naive-Bayes Classifier:")
train_and_print_scores(MultinomialNB(), X, Y)
# use ada-boost classifier
print("AdaBoost Classifier:")
train_and_print_scores(AdaBoostClassifier(), X, Y)

print()
# use tfidf vectorizer for calculating the features
print("Tfidf for features")
print("==================")
X = get_x_from_tfidf_vectorizer()
# use naive bayes classifier
print("Naive-Bayes Classifier:")
train_and_print_scores(MultinomialNB(), X, Y)
# use ada-boost classifier
print("AdaBoost Classifier:")
train_and_print_scores(AdaBoostClassifier(), X, Y)
