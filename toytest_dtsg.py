# Generate toy data to debug the lda negmodel
import numpy as np
import random
from collections import defaultdict
import dtsg
import hofonlineldavb

#import matplotlib.pyplot as plt
#import seaborn as sns

#TODO look at heatmap for p w2|w1
# Should the negmodel learn good topics?

#Does it make sense for the number of docs to be larger than max_vocab
num_docs = 10#0
num_topics = 3
num_neg = 2

max_tokens = 30#100
max_vocab = 10#40

# --- generate data

# Generate beta
topics = np.zeros((num_topics, max_vocab))
for t in range(num_topics):
    non_zero = random.randint(3, 5)#10)
    for i in range(non_zero):
        word_index = random.randint(0, max_vocab-1)
        topics[t][word_index] =  random.randint(3, 10)
    topics[t] = topics[t]/ sum(topics[t])


neg_ids, neg_cnts = [], []
pos_ids, pos_cnts = [], []
#
pos_word_freq = defaultdict(int)
neg_word_freq = defaultdict(int)

theta = np.zeros((num_docs, num_topics))
for d in range(num_docs):
    # Generate theta --  the topic distribution for the document
    non_zero = random.randint(2, 4)
    for i in range(non_zero):
        index = random.randint(0, num_topics - 1)
        theta[d][index] = random.randint(3, 10)
    theta[d] = theta[d] / sum(theta[d])
    print("doc: %d " % d, theta[d])

    #Generate words given the topic distribution
    positive_words = defaultdict(int)
    num_words = random.randint(max_tokens-20, max_tokens)
    for w in range(num_words):
        # Choose a topic given the topic distribution
        z =  np.random.choice(num_topics,  1, p=theta[d])
        word_index = np.random.choice(max_vocab, 1 , p=topics[z[0]])
        positive_words[word_index[0]] += 1
        pos_word_freq[word_index[0]] += 1

    #print(d, " positive words", positive_words)
    pos_ids.append(positive_words.keys())
    pos_cnts.append(positive_words.values())
    #print(pos_ids[-1])
    #print(pos_cnts[-1])

    negative_words = defaultdict(int)
    while sum(negative_words.values()) < (num_words * num_neg):
        neg_index = random.randint(0, max_vocab - 1)
        if neg_index in positive_words.keys():
            continue
        negative_words[neg_index] += 1
        neg_word_freq[word_index[0]] += 1

    #print(d, " negative words", negative_words)
    neg_ids.append(negative_words.keys())
    neg_cnts.append(negative_words.values())
    #print(neg_ids[-1])
    #print(neg_cnts[-1])


vocab, id2vocab = {}, {}
for i in range(max_vocab):
    vocab[str(i)] = i
    id2vocab[i] = str(i)


# Training the negmodel
batch_size = 64
K = num_topics

neggamma = np.zeros((num_docs, K))
print("negative negmodel")
negmodel = dtsg.OnlineLDA(vocab, K=K, D=num_docs, alpha=0.1, eta=0.00000001, zeta=1, tau0=1, kappa=0.5)
i = 0
while i < num_docs:
    nexti = min(i + batch_size, num_docs)
    neggamma[i:nexti], bound = negmodel.update_lambda(pos_ids[i:nexti], pos_cnts[i:nexti], neg_ids[i:nexti], neg_cnts[i:nexti])
    counts_sum = (sum(map(sum, pos_cnts[i:nexti])) + sum(map(sum, neg_cnts[i:nexti])))

    perwordbound = bound * len(pos_ids[i:nexti]) / (num_docs * counts_sum)
    print("perworrd bound" , perwordbound)
    print('%d:  rho_t = %f,  held-out perplexity estimate = %f' % (i, negmodel._rhot, np.exp(-perwordbound)))
    print("approx bound %.5f" % bound)
    i = nexti
    print("number of documents processed: %d" % i)

print("positive negmodel")
posgamma = np.zeros((num_docs, K))
posmodel = hofonlineldavb.OnlineLDA(vocab, K=K, D=num_docs, alpha=0.1, eta=0.01,  tau0=1, kappa=0.5)
i = 0
while i < num_docs:
    nexti = min(i + batch_size, num_docs)
    posgamma[i:nexti], bound = posmodel.update_lambda(pos_ids[i:nexti], pos_cnts[i:nexti])
    counts_sum = sum(map(sum, pos_cnts[i:nexti]))

    perwordbound = bound * len(pos_ids[i:nexti]) / (num_docs * counts_sum)
    print("perworrd bound" , perwordbound)
    print('%d:  rho_t = %f,  held-out perplexity estimate = %f' % (i, posmodel._rhot, np.exp(-perwordbound)))
    print("approx bound %.5f" % bound)
    i = nexti
    print("number of documents processed: %d" % i)

print(topics)
print(theta)
data_joint_prob = np.zeros((max_vocab, max_vocab))
for index_1 in range(max_vocab):
    for index_2 in range(max_vocab):
        print(topics[:][index_1])
        print(theta[index_2][:])

        data_joint_prob[index_1][index_2] = np.dot(topics[:][index_1].T, theta[index_2][:])
print(data_joint_prob)
fig = plt.figure()
fig.subplots_adjust(hspace=0.5)
plt.subplot(311)
ax = sns.heatmap(data_joint_prob, xticklabels=words, yticklabels=words)#, annot_kws=id_word)
plt.xticks(rotation='vertical')
plt.yticks(rotation='horizontal')
plt.title("The data distribution")


for k in range(0, len(negmodel._lambda)):
    lambdak = negmodel._lambda[k]
    denom = (lambdak + negmodel._mu[k])
    lambdak = lambdak / denom
    temp = zip(lambdak, range(0, len(lambdak)))
    temp = sorted(temp, key = lambda x: x[0], reverse=True)
    print('topic %d:' % (k))
    # feel free to change the "53" here to whatever fits your screen nicely.
    for index in range(0, 10):
        print('%20s: n%.4f' % (id2vocab[temp[index][1]], temp[index][0]))
    print()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print("topics of data")
for t in range(num_topics):
    print(topics[t])
print()
for t in range(num_topics):
    print(negmodel._lambda[t]/(negmodel._lambda[t] + negmodel._mu[t]))
print()
for t in range(num_topics):
    print( posmodel._lambda[t] /  sum(posmodel._lambda[t]))

for w in range(num_docs):
    print(theta[w])
    print(neggamma[w] / sum(neggamma[w]))
    print(posgamma[w] / sum(posgamma[w]))
    print()

#ldalambda = posmodel._lambda /  sum(posmodel._lambda)


