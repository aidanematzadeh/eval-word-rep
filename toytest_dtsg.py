# Generate toy data to debug the lda model
import numpy as np
import random
from collections import defaultdict

#import matplotlib.pyplot as plt
#import seaborn as sns

num_docs = 100
num_topics = 5
num_neg = 2

max_tokens = 100
max_vocab = 40

# --- generate data
topics = np.zeros((num_topics, max_vocab))
for t in range(num_topics):
    non_zero = random.randint(3, 10)
    for i in range(non_zero):
        word_index = random.randint(0, max_vocab-1)
        topics[t][word_index] =  random.randint(3, 10)
    topics[t] = topics[t]/ sum(topics[t])

print("topics")
for t in range(num_topics):
    print(t, topics[t], len(topics[t]))

negative_ids, negative_counts = [], []
positive_ids, postive_counts = [], []

theta = np.zeros((num_docs, num_topics))
for d in range(num_docs):
    # Generate the topic distribution for the document
    non_zero = random.randint(2, 4)
    for i in range(non_zero):
        index = random.randint(0, num_topics - 1)
        theta[d][index] = random.randint(3, 10)
    theta[d] = theta[d] / sum(theta[d])
    print("doc: %d " % d, theta[d])

    #Generate words given the topic distribution
    positive_words = defaultdict(int)
    num_words = random.randint(60, max_tokens)
    for w in range(num_words):
        # Choose a topic given the topic distribution
        z =  np.random.choice(num_topics,  1, p=theta[d])
        word_index = np.random.choice(max_vocab, 1 , p=topics[z[0]])
        positive_words[word_index[0]] += 1

    print(d, " positive words", positive_words)
    negative_words = defaultdict(int)
    while sum(negative_words.values()) < (num_words * num_neg):
        neg_index = random.randint(0, max_vocab - 1)
        if neg_index in positive_words.keys():
            continue
        negative_words[neg_index] += 1

    print(d, " negative words", negative_words)


