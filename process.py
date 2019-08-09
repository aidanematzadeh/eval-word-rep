import os

import joblib

import gensim
import numpy as np

import scipy
import scipy.spatial
import scipy.io
import os.path
import codecs
import pandas
import csv
import glob
import pickle
import memory

# Reads and process data used in the evaluation.


class Glove_model(object):
    def __init__(self, glove_path):
        self.df = pandas.read_table(glove_path,sep = ' ', index_col=0,
                                    encoding='utf-8', quoting=csv.QUOTE_NONE)

        # make this into a faster to access object
        self.activation = {}
        for index, row in self.df.iterrows():
            self.activation[index] = row

        self.idx = 0
        self.vocab = set(self.df.index.tolist())

    def similarity(self, word1, word2):
        vector1 = self.activation[word1]
        vector2 = self.activation[word2]
        sim = 1 - scipy.spatial.distance.cosine(vector1, vector2)
        return(sim)


def load_scores(path):
    """ Loads a pickle file.
    """
    print('Loading pickle from: '+path)
    with open(path, 'rb') as f:
        scores = pickle.load(f)
    return scores


def get_norms(norms_pickle, norms_path=None, regeneratePickle=True):
    """ Read Nelson norms for the evaluation methods.
    If a pickle file exists, load and return the norms. Otherwise, read the
    norms from the dir and write a pickle file to norms_pickle.
    Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
    """
    if os.path.exists(norms_pickle) and not regeneratePickle:
        print('Loading an existing norms pickle')
        return load_scores(norms_pickle)

    else:
        print('Regenerating norms pickle...')
        # The value of zero means that the norms[cue][target] does not exist.
        norms = {}
        for filename in glob.glob(os.path.join(norms_path, '*.bin')):
            normfile = codecs.open(filename, 'r', encoding="ISO-8859-1")
            normfile.readline()
            for line in normfile:
                nodes = line.strip().split(',')
                cue = nodes[0].strip().lower()
                target = nodes[1].strip().lower()
                if cue not in norms:
                    norms[cue] = {}
                norms[cue][target] = float(nodes[5])  # FSG, p(target|cue)

        with open(norms_pickle, 'wb') as output:
            joblib.dump(norms, output)
        return norms


def get_w2v(w2vcos_pickle, w2vcond_pickle,
            norms=None, w2v_path=None, flavor=None, cond_eq=None, writePickle=False, regeneratePickle=False):
    """ Load (Gensim) Word2Vec representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """    
    if os.path.exists(w2vcos_pickle) and not regeneratePickle:
        print('Existing W2V pickle found...')
        w2v_cos = load_scores(w2vcos_pickle)
        w2v_cond = load_scores(w2vcond_pickle)
        return w2v_cos, w2v_cond

    if flavor == 'keyed_binary':  # Loading a pretrained binary file from Google
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,
                                                                binary=True)
    elif flavor == 'keyed_text': 
        model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path,
                                                                binary=False)    
    elif flavor == 'gensim':  # Loading a model trained by gensim
        #with open(w2v_path, 'rb') as f:
        #    model = joblib.load(f.read())
        #with open(w2v_path, 'wb') as output:
        #    joblib.dump(model, output)
        model = gensim.models.Word2Vec.load(w2v_path)
    else:
        raise ValueError('Flavor not recognized')    

    print("Done loading the Gensim model.")
    
    def getVocabSize(model, flavor):
        '''two different ways of representing the vocabulary'''
        print('Vocab contains '+str(len(model.vocab.keys()))+' words')
    
    getVocabSize(model, flavor)

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def cueMissing(model, cue, flavor):
        '''two different ways of representing the vocabulary'''
        if flavor == 'gensim':
            return(cue not in model)    
        else:
            return(cue not in model.vocab)

    w2v_cos, w2v_cond = {}, {}
    # List of all the norms in the model. Used in normalization of cond prob.
    wordlist = set([])
    # Note that the cosine is the same as dot product for cbow vectors
    print('Getting cosine similiarities')
    num_cues = 0 
    num_missing_cues = 0

    for cue in norms:
        num_cues += 1 
        if cueMissing(model, cue, flavor):
            num_missing_cues +=1 
            #import pdb
            #pdb.set_trace()
            #print('Cue not found: '+cue)
            continue
        if cue not in w2v_cos:
            w2v_cos[cue], w2v_cond[cue] = {}, {}
        wordlist.add(cue)

        targetlist = set(list(norms[cue].keys()) + list(norms.keys()))
        for target in targetlist:
            if target not in model:
                continue
            if target not in w2v_cos:
                w2v_cos[target], w2v_cond[target] = {}, {}
            if target not in w2v_cos[cue]:
                w2v_cos[cue][target] = np.round(model.similarity(cue, target), decimals=6)
                w2v_cos[target][cue] = w2v_cos[cue][target]

    print('Missing '+str(num_missing_cues)+' of '+str(num_cues)+' cues')

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    print('Getting conditional probability similiarities')
    if cond_eq == 'eq1':
        for cue in w2v_cos.keys():
            cue_context = []
            for w in wordlist:
                cue_context.append(model.similarity(cue, w))
            # Using words in the word_list to normalize the prob
            p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
            for target in w2v_cos[cue]:
                w2v_cond[cue][target] = np.round(np.exp(w2v_cos[cue][target] - p_cue), decimals=6)

    elif cond_eq == 'eq4':
        print('Normalizing to support conditional probability calculations')
        model.column_normalized = np.apply_along_axis(softmax, axis=0,
                                                  arr=model.syn0)
        model.row_normalized= np.apply_along_axis(softmax, axis=1, arr=model.syn0)

        word2id = dict(zip(model.vocab, range(len(model.vocab))))
        #id2word = dict(zip(word2id.values(), word2id.keys()))
        #num_topics = model.row_normalized.shape[1]
        for cue in w2v_cos.keys():
            cue_topics_dist = model.row_normalized[word2id[cue], :]
            for target in w2v_cos[cue]:
                target_probvec = model.column_normalized[word2id[target], :]
                w2v_cond[cue][target] = np.round(np.sum(cue_topics_dist * target_probvec), decimals=6)
    else:
        raise ValueError('Unrecognized equation for conditional probability assessment')

    if writePickle:
        print('Pickling model scores')
        with open(w2vcond_pickle, 'wb') as output:
            joblib.dump(w2v_cond, output)
        with open(w2vcos_pickle, 'wb') as output:
            joblib.dump(w2v_cos, output)
    else:
        print('Not caching pickles because of size')

    print("cosine size %d norms size %d cond size %d" %
          (len(w2v_cos), len(norms), len(w2v_cond)))

    # check the size of the objects in memory    
    return w2v_cos, w2v_cond


def condprob_gsteq8(norms, word2id, topics):
    """
    Griffiths et al eq 8
    p(w2|w1) = sum_z p(w2|z)p(z|w1), 
    p(z|w1) = p(w1|z)p(z)/p(w1)
    """
    condprob = {}
    for cue in norms:
        if cue not in word2id.keys():
            continue
        if cue not in condprob:
            condprob[cue] = {}
        cueid = word2id[cue]
        # Calculate the cond prob for all the targets given the cue, and
        # also all the possible cues
        
        # p(target|cue) = sum_z p(target|z)p(z|cue),
        # p(z|cue) = p(cue|z)p(z)/p(cue)

        targetlist = set(list(norms[cue].keys()) + list(norms.keys()))
        for target in targetlist:
            if target not in word2id.keys():
                continue
            targetid = word2id[target]
            # p(w1) = sum_z p(w1|z)
            # target_prob =  np.sum(topics[:, targetid])
            cue_prob = np.sum(topics[:, cueid])

            condprob[cue][target] = np.dot(topics[:, cueid], topics[:, targetid]) / cue_prob
            #target_prob
            
            # Probability of the topic P(z)
            condprob[cue][target] /= len(topics[:,cueid]) 

        assert sum(condprob[cue].values()) < 1

    return condprob


def condprob_nmgeq4(norms, word2id, topics, gamma):
    condprob = {}
    for cue in norms:
        if cue not in word2id.keys():
            continue
        if cue not in condprob:
            condprob[cue] = {}

        # Topic distribution of the document associated with cue
        cueid = word2id[cue]
        cue_topics_dist = gamma[cueid] / sum(gamma[cueid])  # Normalize gamma
        # Calculate the cond prob for all the targets given the cue, and
        # also all the possible cues
        targetlist = set(list(norms[cue].keys()) + list(norms.keys()))
        for target in targetlist:
            if target not in word2id.keys():
                continue
            targetid = word2id[target]
            condprob[cue][target] = np.dot(cue_topics_dist, topics[:, targetid])
    return condprob


def get_gibbslda_avg(gibbslda_pickle, beta=0.01, norms=None, vocab_path=None, lambda_path=None, writePickle=False, regeneratePickle=False):
    """ Get the cond prob for word representations learned by
    Gibbs sampler code.
    vocab_path: the word2id mappings
    """

    if os.path.exists(gibbslda_pickle) and not regeneratePickle:
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
        num_topics = topics.shape[0]

        # p(target|cue) = sum_z p(target|z)p(z|cue),
        # p(z|cue) = p(cue|z)(z)/p(cue)

        for k in range(num_topics):
            topics[k] = topics[k] / sum(topics[k])
        condprobs[filename] = condprob_gsteq8(norms, word2id, topics)
        count += 1
        memory.memoryCheckpoint(filename,'LDA_load')

    print('Getting conditional probability similiarities')
    avg_condprob = {}
    for filename in condprobs:
        for cue in condprobs[filename]:
            if cue not in avg_condprob:
                avg_condprob[cue] = {}
            for target in condprobs[filename][cue]:
                if target not in avg_condprob[cue]:
                    avg_condprob[cue][target] = 0
                avg_condprob[cue][target] += condprobs[filename][cue][target]

    for cue in avg_condprob:
        for target in avg_condprob[cue]:
            avg_condprob[cue][target] /= len(condprobs.keys())

    if writePickle:
        with open(gibbslda_pickle, 'wb') as output:
            joblib.dump(avg_condprob, output)

    return avg_condprob




def get_gibbslda(gibbslda_path, beta=0.01, norms=None, vocab_path=None,
            lambda_path=None, regeneratePickle = False):
    """ Get the cond prob for word representations learned by
    Gibbs sampler code.
    vocab_path: the word2id mappings
    """

    if os.path.exists(gibbslda_path) and not regeneratePickle:
        return load_scores(gibbslda_path)

    #TODO need to change 1-->0?
    word2id, id2word = read_tsgvocab(vocab_path)

    # Getting the topic-word probs -- p(w|topic)
    # import scipy.io
    # topics = scipy.io.loadmat(lambda_path)['wp'].T + beta
    topics = np.loadtxt(lambda_path).T + beta
    print("lambda", topics.shape)
    num_topics = topics.shape[0]

    # p(target|cue) = sum_z p(target|z)p(z|cue),
    # p(z|cue) = p(cue|z)(z)/p(cue)

    for k in range(num_topics):
        topics[k] = topics[k] / sum(topics[k])
    condprob = condprob_gsteq8(norms, word2id, topics)


    with open(gibbslda_path, 'wb') as output:
        joblib.dump(condprob, output)

    return condprob


def get_tsg(tsg_path, cond_eq, norms=None, vocab_path=None,
            lambda_path=None, gamma_path=None, mu_path=None, regeneratePickle=False):
    """ Get the cond prob for word representations learned by
    Hoffman-VBLDA-based code.
    Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
    p(topic|cue) = theta_cue[topic] because document is the cue
    vocab_path: the word2id mappings
    """

    if os.path.exists(tsg_path) and not regeneratePickle:
        return load_scores(tsg_path)

    word2id, id2word = read_tsgvocab(vocab_path)

    # Getting the topic-word probs -- p(w|topic)
    topics = np.loadtxt(lambda_path)
    print("lambda", topics.shape)
    num_topics = topics.shape[0]

    # p(w2|w1) = sum_z p(w2|z)p(z|w1), p(z|w1) = p(w1|z)(z)/p(w1)
    if cond_eq == "gst-eq8":
       condprob = condprob_gsteq8(norms, word2id, topics)

    if cond_eq == "nmg-eq4":
        gamma = np.loadtxt(gamma_path)  # p(topic|doc)
#        print("number of topics %d gamma %d lambda %d" % (num_topics, gamma.shape, topics.shape))
        print("gamma", gamma.shape)

        # Normalize the topic-word probs
        if mu_path is None:
            for k in range(num_topics):
                topics[k] = topics[k] / sum(topics[k])
        else:
            mu = (np.loadtxt(mu_path))
            for k in range(num_topics):
                denom = topics[k] + mu[k]
                topics[k] = topics[k] / denom

        condprob = condprob_nmgeq4(norms, word2id, topics, gamma)

    with open(tsg_path, 'wb') as output:
        joblib.dump(condprob, output)

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
            w, wid, wfreq = line.split()
            word2id[w] = int(wid)
            id2word[int(wid)] = w
    return word2id, id2word


def get_tsgfreq(tsgfreq_pickle, norms=None, vocab_path=None,
                counts_path=None, ids_path=None, writePickle=False, regeneratePickle= False):
    """ Get the freq of each word in the documents in TSG model.
    vocab_path: the word2id mappings
    """
    if os.path.exists(tsgfreq_pickle) and not regeneratePickle:
        return load_scores(tsgfreq_pickle)

    ids, counts = read_tsgdata(counts_path, ids_path) #this appears to be limited to items in the norms
    #!!! asking Aida about this
    word2id, id2word = read_tsgvocab(vocab_path)

    tsgfreq = {}
    for cue in norms:
        #print('cue: '+cue)
        if cue not in word2id.keys():
            continue
        if cue not in tsgfreq:
            tsgfreq[cue] = {}

        cueid = word2id[cue]

        targetlist = set(list(norms[cue].keys()) + list(norms.keys()))
        for targetid, targetcount in zip(ids[cueid], counts[cueid]):      
            #print(targetid)
            target = id2word[targetid]
            #print('target: '+target)
            if target not in targetlist:
                continue
            else:
                tsgfreq[cue][target] = targetcount

        # TODO some of the targets do not happen in the document,
        # their freq is zero.
        for target in targetlist:
            if target not in word2id.keys():
                continue
            if target not in tsgfreq[cue].keys():
                tsgfreq[cue][target] = 1

    if writePickle:
        with open(tsgfreq_pickle, 'wb') as output:
            joblib.dump(tsgfreq, output)

    return tsgfreq



def get_allpairs(allpairs_pickle, norms, cbow=None, sg=None, lda=None, glove=None, regeneratePickle=False):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_pickle) and not regeneratePickle:
        print('Loading allpairs froma  pickle')
        return load_scores(allpairs_pickle)

    print('Regenerating allpairs')
    allpairs, normpairs = [], []
    for cue in norms:
        for target in norms[cue]:
            normpairs.append((cue, target))
            if (cbow is not None) and\
                    ((cue not in cbow) or (target not in cbow[cue])):
                continue

            if (sg is not None) and\
                    ((cue not in sg) or (target not in sg[cue])):
                continue

            if (lda is not None) and\
                    ((cue not in lda) or (target not in lda[cue])):
                continue

            if (glove is not None) and\
                    ((cue not in glove) or (target not in glove[cue])):
                continue

            allpairs.append((cue, target))

    print("cues and targets in norms %d" % len(normpairs))
    print("cues and targets in norms and other data %d" % len(allpairs))

    with open(allpairs_pickle, 'wb') as output:
        joblib.dump(allpairs, output)
    return allpairs

def get_allpairs_generalized(allpairs_cache_path, norms, models, regeneratePickle=False):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_cache_path) and not regeneratePickle:
        return load_scores(allpairs_cache_path)
        
    allpairs, normpairs = [], []
    for cue in norms:
        for target in norms[cue]:
            normpairs.append((cue, target))
            cue_present = np.array([0 if cue not in model else 1 for model in models])
            target_present = np.array([0 if target not in model else 1 for model in models])
            
            if not np.all(cue_present):
                num_cues_missing = len(cue_present) - np.sum(cue_present)
                print('Missing cue: '+cue+" (missing from "+str(num_cues_missing)+" models)")
            elif not np.all(target_present):
                num_targets_missing = len(target_present) - np.sum(target_present)
                print('Missing cue: '+target+" (missing from "+str(num_targets_missing)+" models)")
            else:
                allpairs.append((cue, target))
                        

    print("cues and targets in evaluation dataset: %d" % len(normpairs))
    print("cues and targets in evaluation dataset and all models: %d" % len(allpairs))

    with open(allpairs_cache_path, 'wb') as output:
        joblib.dump(allpairs, output)
    return allpairs


def get_pair_scores(scores, allpairs):
    """ Return the cue-target scores
    """
    pair_scores = []
    for cue, target in allpairs:
        pair_scores.append(scores[cue][target])
    return pair_scores


def get_asym_pairs(norms, allpairs):
    """ Return the pairs for which both p(target|cue) and P(cue|target) exist.
    """
    print("asym_pairs_test")
    assym_pairs = set()
    for cue, target in allpairs:
        if not (cue in norms and target in norms[cue]):
            continue
        if not (target in norms and cue in norms[target]):
            continue
        assym_pairs.add((min(cue, target), max(cue, target)))
    return list(assym_pairs)


def get_tuples(tuples_pickle, norms, allpairs, regeneratePickle=False, writeTuple=True):
    """ Find all the three words for which P(w2|w1), P(w3|w2),
    and P(w3|w1) exist.
    This is equivalent to the subset of the combination of 3-length ordered
    tuples that their pairs exist in Nelson norms.
    """
    if os.path.exists(tuples_pickle) and not regeneratePickle:
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
        joblib.dump(tuples, output)

    return tuples


def get_glove(glovecos_pickle, glovecond_pickle, glove_path,
            norms=None, cond_eq=1, writePickle=False, regeneratePickle=False):
    """ Load (Gensim) Word2Vec representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """
    if cond_eq != "eq1":
        raise NotImplementedError

    if os.path.exists(glovecos_pickle) and os.path.exists(glovecond_pickle) and not regeneratePickle:
        print('Existing Glove pickle found...')
        glove_cos = load_scores(glovecos_pickle)
        glove_cond = load_scores(glovecond_pickle)
        return glove_cos, glove_cond

    model =  Glove_model(glove_path)

    glove_cos, glove_cond = {}, {}
    # List of all the norms in the model. Used in normalization of cond prob.
    wordlist = set([])
    # Note that the cosine is the same as dot product for cbow vectors
    print('Getting cosine similarity')
    for cue in norms:
        # print('Getting cosine similarity: '+cue)
        if cue not in model.vocab:
            continue
        if cue not in glove_cos:
            glove_cos[cue], glove_cond[cue] = {}, {}
        wordlist.add(cue)

        targetlist = set(list(norms[cue].keys()) + list(norms.keys()))
        for target in targetlist:
            #print('Checking target: '+target)
            if target not in model.vocab:
                continue
            if target not in glove_cos:
                glove_cos[target], glove_cond[target] = {}, {}
            if target not in glove_cos[cue]:
                glove_cos[cue][target] = model.similarity(cue, target)
                glove_cos[target][cue] = glove_cos[cue][target]

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    print('Getting conditional similarity')
    for cue in glove_cos.keys():
        # print('Getting conditional similarity: '+cue)
        cue_context = []
        for w in wordlist:
            cue_context.append(model.similarity(cue, w))
        # Using words in the word_list to normalize the prob
        p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
        for target in glove_cos[cue]:
            glove_cond[cue][target] = np.exp(glove_cos[cue][target] - p_cue)

    if writePickle:
        print('Writing pickles')
        with open(glovecond_pickle, 'wb') as output:
            joblib.dump(glove_cond, output)
        with open(glovecos_pickle, 'wb') as output:
            joblib.dump(glove_cos, output)

    print("cosine size %d norms size %d cond size %d" %
          (len(glove_cos), len(norms), len(glove_cond)))

    return glove_cos, glove_cond
