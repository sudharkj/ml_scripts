# course: https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class/all_book_titles.txt

# general imports
import matplotlib.pyplot as plt
import nltk
import numpy as np
import sys

# class imports
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('data/latent-semantic-analysis/all_book_titles.txt')]
stopwords = set(w.rstrip() for w in open('data/stopwords.txt'))
stopwords = stopwords.union({
    'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential',
    'printed', 'third', 'second', 'fourth'
})


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
    # remove numbers
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    return tokens


# tokenize the titles and cache them
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        token_array = my_tokenizer(title)
        all_tokens.append(token_array)
        for token in token_array:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except:
        print("Unexpected error:", sys.exc_info())
        pass


# create a normalized vector to each tokenized review
# set the last element to the given label
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        ind = word_index_map[t]
        x[ind] += 1
    return x


# compute data based on the tokens
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for token_array in all_tokens:
    X[:, i] = tokens_to_vector(token_array)
    i += 1

# cluster and plot the results
svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
plt.show()
