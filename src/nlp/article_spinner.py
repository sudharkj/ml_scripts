# course: https://www.udemy.com/data-science-natural-language-processing-in-python
# dataset: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

# general imports
import nltk
import random

# class imports
from bs4 import BeautifulSoup


positive_reviews = BeautifulSoup(open('data/sentiment/electronics/positive.review'))
positive_reviews = positive_reviews.findAll('review_text')

# create trigrams of positive reviews
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])

# calculate the probabilities of the trigrams
trigrams_probabilities = {}
for k, words in trigrams.items():
    # do not consider words with single occurrence
    if len(set(words)) > 1:
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        for w, c in d.items():
            d[w] = float(c) / n
        trigrams_probabilities[k] = d


# pick a random word from the given set of words
def random_sample(word_density):
    r = random.random()
    cumulative = 0
    for word, p in word_density.items():
        cumulative += p
        if r < cumulative:
            return word


# function to generate random similar article
def test_spinner():
    random_review = random.choice(positive_reviews)
    random_review = random_review.text.lower()
    print("Original:", random_review)
    token_array = nltk.tokenize.word_tokenize(random_review)
    for ind in range(len(token_array) - 2):
        if random.random() < 0.2:
            key = (token_array[ind], token_array[ind+2])
            if key in trigrams_probabilities:
                random_word = random_sample(trigrams_probabilities[key])
                token_array[ind+1] = random_word
    print("Spun:")
    print(" ".join(token_array).replace(" .", ".").replace(" '", "'")
          .replace(" ,", ",").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()
