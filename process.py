import csv
import codecs
import glob
import os

from collections import defaultdict

import joblib

import gensim
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.spatial

# Reads and process data used in the evaluation.

class Glove_model(object):
    def __init__(self, glove_path):
        self.df = pd.read_table(glove_path, sep=' ', index_col=0,
                                encoding='utf-8', quoting=csv.QUOTE_NONE)

        # make this into a faster to access object
        self.activation = dict(self.df.iterrows())
        self.idx = 0
        self.vocab = set(self.df.index.tolist())

    def similarity(self, word1, word2):
        vector1 = self.activation[word1]
        vector2 = self.activation[word2]
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)
        return sim


def load_scores(path):
    """ Loads a pickle file.
    """
    with open(path, 'rb') as f:
        scores = joblib.load(f)
    return scores


def get_norms(norms_pickle, norms_path=None, regenerate_pickle=False):
    """ Read Nelson norms for the evaluation methods.
    If a pickle file exists, load and return the norms. Otherwise, read the
    norms from the dir and write a pickle file to norms_pickle.
    Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
    """
    if os.path.exists(norms_pickle) and not regenerate_pickle:
        return load_scores(norms_pickle)

    # The value of zero means that the norms[cue][target] does not exist.
    norms = defaultdict(dict)
    for filename in glob.glob(os.path.join(norms_path, '*.bin')):
        normfile = codecs.open(filename, 'r', encoding="ISO-8859-1")
        normfile.readline()
        for line in normfile:
            nodes = line.strip().split(',')
            cue, target = (node.strip().lower() for node in nodes)
            norms[cue][target] = float(nodes[5])  # FSG, p(target|cue)

    with open(norms_pickle, 'wb') as output:
        joblib.dump(norms, output, protocol=4)
    return norms


def get_w2v(w2vcos_pickle, w2vcond_pickle,
            norms=None, w2v_path=None, binary_flag=False, cond_eq=None, write_pickle=True, regenerate_pickle=False):
    """ Load (Gensim) Word2Vec representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """
    if os.path.exists(w2vcos_pickle) and not regenerate_pickle:
        print('Existing W2V pickle found...')
        w2v_cos = load_scores(w2vcos_pickle)
        w2v_cond = load_scores(w2vcond_pickle)
        return w2v_cos, w2v_cond

    if binary_flag:  # Loading a pretrained binary file from Google
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,
                                                                binary=True)
    else:  # Loading a model trained by gensim
        model = gensim.models.Word2Vec.load(w2v_path)

    print("Done loading the Gensim model.")

    def softmax(x):
        e_x = np.exp(x - x.max(axis=0))
        return e_x / e_x.sum(axis=0)


    w2v_cos, w2v_cond = defaultdict(dict), defaultdict(dict)
    # List of all the norms in the model. Used in normalization of cond prob.
    wordlist = set()
    # Note that the cosine is the same as dot product for cbow vectors
    print('Getting cosine similiarities')
    for cue in norms:
        if cue not in model:
            continue
        wordlist.add(cue)

        targetlist = set(norms[cue]).union(norms).intersection(model)
        for target in targetlist:
            if target not in w2v_cos[cue]:
                w2v_cos[cue][target] = model.similarity(cue, target).round(decimals=6)
                w2v_cos[target][cue] = w2v_cos[cue][target]

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    print('Getting conditional probability similiarities')
    if cond_eq == 'eq1':
        for cue in w2v_cos:
            cue_context = [model.similarity(cue, w) for w in wordlist]
            # Using words in the word_list to normalize the prob
            p_cue = scipy.misc.logsumexp(np.array(cue_context))
            for target in w2v_cos[cue]:
                w2v_cond[cue][target] = np.exp(w2v_cos[cue][target] - p_cue).round(decimals=6)

    elif cond_eq == 'eq4':
        print('Normalizing to support conditional probability calculations')
        model.column_normalized = np.apply_along_axis(softmax, axis=0,
                                                      arr=model.syn0)
        model.row_normalized = np.apply_along_axis(softmax, axis=1, arr=model.syn0)

        word2id = dict(zip(model.vocab, range(len(model.vocab))))
        #id2word = dict(zip(word2id.values(), word2id.keys()))
        #num_topics = model.row_normalized.shape[1]
        for cue in w2v_cos:
            cue_topics_dist = model.row_normalized[word2id[cue], :]
            for target in w2v_cos[cue]:
                target_probvec = model.column_normalized[word2id[target], :]
                w2v_cond[cue][target] = (cue_topics_dist * target_probvec).sum().round(decimals=6)
    else:
        raise ValueError('Unrecognized equation for conditional probability assessment')

    if write_pickle:
        print('Pickling model scores')
        with open(w2vcond_pickle, 'wb') as output:
            joblib.dump(w2v_cond, output, protocol=4)
        with open(w2vcos_pickle, 'wb') as output:
            joblib.dump(w2v_cos, output, protocol=4)
    else:
        print('Not caching pickles because of size')

    print("cosine size %d norms size %d cond size %d" %
          (len(w2v_cos), len(norms), len(w2v_cond)))

    return w2v_cos, w2v_cond


def condprob_gsteq8(norms, word2id, topics):
    """
    Griffiths et al eq 8
    p(w2|w1) = sum_z p(w2|z)p(z|w1),
    p(z|w1) = p(w1|z)p(z)/p(w1)
    """
    condprob = defaultdict(dict)
    for cue in norms:
        if cue not in word2id:
            continue
        cueid = word2id[cue]
        # Calculate the cond prob for all the targets given the cue, and
        # also all the possible cues
        
        # p(target|cue) = sum_z p(target|z)p(z|cue),
        # p(z|cue) = p(cue|z)p(z)/p(cue)

        targetlist = set(norms[cue]).union(norms).intersection(word2id)
        for target in targetlist:
            targetid = word2id[target]
            # p(w1) = sum_z p(w1|z)
            # target_prob = topics[:, targetid].sum()
            cue_prob = topics[:, cueid].sum()

            condprob[cue][target] = topics[:, cueid].dot(topics[:, targetid]) / cue_prob
            #target_prob

            # Probability of the topic P(z)
            condprob[cue][target] /= len(topics[:, cueid])

        assert sum(condprob[cue].values()) < 1

    return condprob


def condprob_nmgeq4(norms, word2id, topics, gamma):
    condprob = defaultdict(dict)
    for cue in norms:
        if cue not in word2id:
            continue

        # Topic distribution of the document associated with cue
        cueid = word2id[cue]
        cue_topics_dist = gamma[cueid] / sum(gamma[cueid])  # Normalize gamma
        # Calculate the cond prob for all the targets given the cue, and
        # also all the possible cues
        targetlist = set(norms[cue]).union(norms).intersection(word2id)
        for target in targetlist:
            targetid = word2id[target]
            condprob[cue][target] = cue_topics_dist.dot(topics[:, targetid])
    return condprob


def get_gibbslda_avg(gibbslda_pickle, beta=0.01, norms=None, vocab_path=None, lambda_path=None, write_pickle=True, regenerate_pickle=False):
    """ Get the cond prob for word representations learned by
    Gibbs sampler code.
    vocab_path: the word2id mappings
    """

    if os.path.exists(gibbslda_pickle) and not regenerate_pickle:
        print('Existing Gibbs sampling LDA pickle found...')
        return load_scores(gibbslda_pickle)

    word2id, id2word = read_tsgvocab(vocab_path)

    # Getting the topic-word probs -- p(w|topic)

    print('Loading LDA samples')
    condprobs = {}
    count = 0
    for filename in os.listdir(lambda_path):
        print(filename)
        topics = scipy.io.loadmat(lambda_path+filename)['wp'].todense().T + beta
        topics = np.asarray(topics)  # np.loadtxt(lambda_path).T + beta
        print("lambda", topics.shape)

        # p(target|cue) = sum_z p(target|z)p(z|cue),
        # p(z|cue) = p(cue|z)(z)/p(cue)

        topics /= topics.sum(axis=1, keepdims=True)
        condprobs[filename] = condprob_gsteq8(norms, word2id, topics)
        count += 1

    print('Getting conditional probability similiarities')
    avg_condprob = defaultdict(dict)
    for filename in condprobs:
        for cue in condprobs[filename]:
            for target in condprobs[filename][cue]:
                if target not in avg_condprob[cue]:
                    avg_condprob[cue][target] = 0
                avg_condprob[cue][target] += condprobs[filename][cue][target]

    for cue in avg_condprob:
        for target in avg_condprob[cue]:
            avg_condprob[cue][target] /= len(condprobs)

    if write_pickle:
        with open(gibbslda_pickle, 'wb') as output:
            joblib.dump(avg_condprob, output, protocol=4)

    return avg_condprob




def get_gibbslda(gibbslda_path, beta=0.01, norms=None, vocab_path=None,
            lambda_path=None):
    """ Get the cond prob for word representations learned by
    Gibbs sampler code.
    vocab_path: the word2id mappings
    """

    if os.path.exists(gibbslda_path):
        return load_scores(gibbslda_path)

    #TODO need to change 1-->0?
    word2id, id2word = read_tsgvocab(vocab_path)

    # Getting the topic-word probs -- p(w|topic)
    # import scipy.io
    # topics = scipy.io.loadmat(lambda_path)['wp'].T + beta
    topics = np.loadtxt(lambda_path).T + beta
    print("lambda", topics.shape)

    # p(target|cue) = sum_z p(target|z)p(z|cue),
    # p(z|cue) = p(cue|z)(z)/p(cue)

    topics /= topics.sum(axis=1, keepdims=True)
    condprob = condprob_gsteq8(norms, word2id, topics)

    with open(gibbslda_path, 'wb') as output:
        joblib.dump(condprob, output, protocol=4)

    return condprob


def get_tsg(tsg_path, cond_eq, norms=None, vocab_path=None,
            lambda_path=None, gamma_path=None, mu_path=None):
    """ Get the cond prob for word representations learned by
    Hoffman-VBLDA-based code.
    Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
    p(topic|cue) = theta_cue[topic] because document is the cue
    vocab_path: the word2id mappings
    """

    if os.path.exists(tsg_path):
        return load_scores(tsg_path)

    word2id, id2word = read_tsgvocab(vocab_path)

    # Getting the topic-word probs -- p(w|topic)
    topics = np.loadtxt(lambda_path)
    print("lambda", topics.shape)

    # p(w2|w1) = sum_z p(w2|z)p(z|w1), p(z|w1) = p(w1|z)(z)/p(w1)
    if cond_eq == "gst-eq8":
       condprob = condprob_gsteq8(norms, word2id, topics)

    if cond_eq == "nmg-eq4":
        gamma = np.loadtxt(gamma_path)  # p(topic|doc)
#        print("number of topics %d gamma %d lambda %d" % (topics.shape[0], gamma.shape, topics.shape))
        print("gamma", gamma.shape)

        # Normalize the topic-word probs
        if mu_path is None:
            topics /= topics.sum(axis=1, keepdims=True)
        else:
            mu = np.loadtxt(mu_path)
            topics /= topics + mu

        condprob = condprob_nmgeq4(norms, word2id, topics, gamma)

    with open(tsg_path, 'wb') as output:
        joblib.dump(condprob, output, protocol=4)

    return condprob


def read_tsgdata(counts_path, ids_path):
    # Reading the word ids and counts
    idfile = open(ids_path, 'r')
    countfile = open(counts_path, 'r')
    ids, counts = [], []
    for index, (idline, ctline) in enumerate(zip(idfile, countfile)):
        # assert index == int(idline.split()[0].strip(':'))
        ids.append([int(wid) for wid in idline.split()[1:]])
        counts.append([int(wcount) for wcount in ctline.split()[1:]])
    return ids, counts


def read_tsgvocab(vocab_path):
    word2id, id2word = {}, {}
    with open(vocab_path, 'r') as f:
        for line in f:
            w, wid, _ = line.split()
            word2id[w] = int(wid)
            id2word[int(wid)] = w
    return word2id, id2word


def get_tsgfreq(tsgfreq_pickle, norms=None, vocab_path=None,
                counts_path=None, ids_path=None, write_pickle=True, regenerate_pickle= False):
    """ Get the freq of each word in the documents in TSG model.
    vocab_path: the word2id mappings
    """
    if os.path.exists(tsgfreq_pickle) and not regenerate_pickle:
        return load_scores(tsgfreq_pickle)

    ids, counts = read_tsgdata(counts_path, ids_path)
    word2id, id2word = read_tsgvocab(vocab_path)

    tsgfreq = defaultdict(dict)
    for cue in norms:
        if cue not in word2id:
            continue

        cueid = word2id[cue]

        targetlist = set(norms[cue]).union(norms)
        for targetid, targetcount in zip(ids[cueid], counts[cueid]):
            target = id2word[targetid]
            if target not in targetlist:
                continue
            tsgfreq[cue][target] = targetcount

        # TODO some of the targets do not happen in the document,
        # their freq is zero.
        targetlist = targetlist.intersection(word2id)
        for target in targetlist:
            if target not in tsgfreq[cue]:
                tsgfreq[cue][target] = 1

    if write_pickle:
        with open(tsgfreq_pickle, 'wb') as output:
            joblib.dump(tsgfreq, output, protocol=4)

    return tsgfreq


def get_allpairs(allpairs_pickle, norms, cbow=None, sg=None, lda=None, glove=None, regenerate_pickle=False):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_pickle) and not regenerate_pickle:
        return load_scores(allpairs_pickle)

    allpairs, normpairs = [], []
    for cue in norms:
        for target in norms[cue]:
            normpairs.append((cue, target))
            if cbow and (cue not in cbow or target not in cbow[cue]):
                continue

            if sg and (cue not in sg or target not in sg[cue]):
                continue

            if lda and (cue not in lda or target not in lda[cue]):
                continue

            if glove and (cue not in glove or target not in glove[cue]):
                continue

            allpairs.append((cue, target))

    print("cues and targets in norms %d" % len(normpairs))
    print("cues and targets in norms and other data %d" % len(allpairs))

    with open(allpairs_pickle, 'wb') as output:
        joblib.dump(allpairs, output, protocol=4)
    return allpairs

def get_allpairs_generalized(allpairs_pickle, norms, models, regenerate_pickle=False):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_pickle) and not regenerate_pickle:
        return load_scores(allpairs_pickle)

    allpairs, normpairs = [], []
    for cue in norms:
        for target in norms[cue]:
            normpairs.append((cue, target))
            model_presence = [cue in model and target in model[cue] for model in models]
            #print(model_presence)
            if all(model_presence):
                allpairs.append((cue, target))
            # elif sum(model_presence) == (len(model_presence) -1):
                #import pdb
                #pdb.set_trace()
            else:
                #import pdb
                #pdb.set_trace()
                print('Missing cue or target!')
                print((cue, target))

    print("cues and targets in norms %d" % len(normpairs))
    print("cues and targets in norms and other data %d" % len(allpairs))

    with open(allpairs_pickle, 'wb') as output:
        joblib.dump(allpairs, output, protocol=4)
    return allpairs


def get_pair_scores(scores, allpairs):
    """ Return the cue-target scores
    """
    return [scores[cue][target] for cue, target in allpairs]


def get_asym_pairs(norms, allpairs):
    """ Return the pairs for which both p(target|cue) and P(cue|target) exist.
    """
    asym_pairs = set()
    for cue, target in allpairs:
        if cue not in norms or target not in norms[cue]:
            continue
        if target not in norms or cue not in norms[target]:
            continue
        asym_pairs.add((min(cue, target), max(cue, target)))
    return asym_pairs


def get_tuples(tuples_pickle, norms, allpairs, regenerate_pickle=False, writeTuple=True):
    """ Find all the three words for which P(w2|w1), P(w3|w2),
    and P(w3|w1) exist.
    This is equivalent to the subset of the combination of 3-length ordered
    tuples that their pairs exist in Nelson norms.
    """
    if os.path.exists(tuples_pickle) and not regenerate_pickle:
        return load_scores(tuples_pickle)

    # TODO make faster
    allpairs = set(allpairs)
    tuples = []
    for w1 in norms:
        for w2 in norms[w1]:
            if w2 not in norms:
                continue
            if (w1, w2) not in allpairs:
                continue
            for w3 in norms[w2]:
                if (w2, w3) not in allpairs:
                    continue
                if (w1, w3) in allpairs:
                    tuples.append((w1, w2, w3))

    with open(tuples_pickle, 'wb') as output:
        joblib.dump(tuples, output, protocol=4)

    return tuples


def get_glove(glovecos_pickle, glovecond_pickle, glove_path,
            norms=None, cond_eq=1, write_pickle=True, regenerate_pickle=False):
    """ Load (Gensim) Word2Vec representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """
    if cond_eq != "eq1":
        raise NotImplementedError

    if os.path.exists(glovecos_pickle) and os.path.exists(glovecond_pickle) and not regenerate_pickle:
        print('Existing Glove pickle found...')
        glove_cos = load_scores(glovecos_pickle)
        glove_cond = load_scores(glovecond_pickle)
        return glove_cos, glove_cond

    model = Glove_model(glove_path)

    glove_cos, glove_cond = defaultdict(dict), defaultdict(dict)
    # List of all the norms in the model. Used in normalization of cond prob.
    wordlist = set()
    # Note that the cosine is the same as dot product for cbow vectors
    print('Getting cosine similarity')
    for cue in norms:
        # print('Getting cosine similarity: '+cue)
        if cue not in model.vocab:
            continue
        wordlist.add(cue)

        targetlist = set(norms[cue]).union(norms).intersection(model.vocab)
        for target in targetlist:
            if target not in glove_cos[cue]:
                glove_cos[cue][target] = model.similarity(cue, target)
                glove_cos[target][cue] = glove_cos[cue][target]

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    print('Getting conditional similarity')
    for cue in glove_cos:
        # print('Getting conditional similarity: '+cue)
        cue_context = [model.similarity(cue, w) for w in wordlist]
        # Using words in the word_list to normalize the prob
        p_cue = scipy.misc.logsumexp(np.array(cue_context))
        for target in glove_cos[cue]:
            glove_cond[cue][target] = np.exp(glove_cos[cue][target] - p_cue)

    if write_pickle:
        print('Writing pickles')
        with open(glovecond_pickle, 'wb') as output:
            joblib.dump(glove_cond, output, protocol=4)
        with open(glovecos_pickle, 'wb') as output:
            joblib.dump(glove_cos, output, protocol=4)

    print("cosine size %d norms size %d cond size %d" %
          (len(glove_cos), len(norms), len(glove_cond)))

    return glove_cos, glove_cond
