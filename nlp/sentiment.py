# course: https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

# general imports
import nltk
import numpy as np

# class imports
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression


wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('data/stopwords.txt'))

positive_reviews = BeautifulSoup(open('data/sentiment/electronics/positive.review'))
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('data/sentiment/electronics/negative.review'))
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
# have equal number of positive and negative reviews for better modeling
if len(positive_reviews) > len(negative_reviews):
    positive_reviews = positive_reviews[:len(negative_reviews)]
else:
    negative_reviews = negative_reviews[:len(positive_reviews)]

word_index_map = {}
current_index = 0


# custom tokenizer
def my_tokenizer(s):
    # make a case insensitive modeling
    s = s.lower()
    # use nltk tokenize instead of normal split
    tokens = nltk.tokenize.word_tokenize(s)
    # token with single occurence might not be of any use
    tokens = [t for t in tokens if len(t) > 2]
    # use lemmatized tokens
    # add them to array even if they exist because the count matters
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    # remove stopwords from the tokens
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


# tokenize the reviews while assigning index to each new token
def get_tokenized_reviews(reviews):
    global current_index
    tokenized_reviews = []
    for review in reviews:
        tokens = my_tokenizer(review.text)
        # cache the tokenized reviews
        tokenized_reviews.append(tokens)
        # assign index to each unique token
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
    return tokenized_reviews


positive_tokenized = get_tokenized_reviews(positive_reviews)
negative_tokenized = get_tokenized_reviews(negative_reviews)

# same as 2*len(positive_reviews) since they are of equal length
N = len(positive_reviews) + len(negative_reviews)

data = np.zeros((N, len(word_index_map) + 1))
i = 0


# create a normalized vector to each tokenized review
# set the last element to the given label
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        ind = word_index_map[t]
        x[ind] += 1
    x = x/x.sum()
    x[-1] = label
    return x


# set data based on reviews
def set_data_based_on_reviews(reviews_tokenized, label):
    global i  # to include both positive and negative reviews
    for tokens in reviews_tokenized:
        xy = tokens_to_vector(tokens, label)
        data[i, :] = xy
        i += 1


set_data_based_on_reviews(positive_tokenized, 1)
set_data_based_on_reviews(negative_tokenized, 0)

# shuffle data and separate feature-vectors from labels
np.random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1]

# divide the data to train and test set
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

# train the model and print the score
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))

# see words with higher impact
threshold = 0.5
for word, index in word_index_map.items():
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
