import os
import gensim
import numpy as np
import scipy.io as sio
import scipy
import pickle
import os.path
# Reads and process data used in the evaluation.
from smart_open import smart_open

def load_scores(path):
    """ Loads a pickle file.
    """
    with open(path, 'rb') as f:
        scores = pickle.load(f)
    return scores


def get_norms(norms_pickle, norms_path=None):
    """ Read Nelson norms for the evaluation methods.
    If a pickle file exists, load and return the norms. Otherwise, read the
    norms from the dir and write a pickle file to norms_pickle.
    Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
    """
    if os.path.exists(norms_pickle):
        return load_scores(norms_pickle)

    # The value of zero means that the norms[cue][target] does not exist.
    norms = {}
    for filename in os.listdir(norms_path):
        normfile = open(norms_path + "/" + filename, 'r', encoding="ISO-8859-1")
        normfile.readline()
        for line in normfile:
            nodes = line.strip().split(',')
            cue = nodes[0].strip().lower()
            target = nodes[1].strip().lower()
            if cue not in norms:
                norms[cue] = {}
            norms[cue][target] = float(nodes[5])  # FSG, p(target|cue)

    with open(norms_pickle, 'wb') as output:
        pickle.dump(norms, output)
    return norms


def get_w2v(w2vcos_pickle, w2vcond_pickle,
            norms=None, w2v_path=None, binary_flag=None):
    """ Load (Gensim) Word2Vec representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """
    if os.path.exists(w2vcos_pickle):
        w2v_cos = load_scores(w2vcos_pickle)
        w2v_cond = load_scores(w2vcond_pickle)
        return w2v_cos, w2v_cond

    if binary_flag:  # Loading a pretrained binary file from Google
        model = gensim.models.Word2Vec.load_word2vec_format(w2v_path,
                                                            binary=True)
    else:  # Loading a model trained by gensim
        #with smart_open(w2v_path, 'rb') as f:
            #u = pickle._Unpickler(f)
            #u.encoding = 'latin1'
            #p = u.load()
            #print(p)
        #    print(w2v_path)
        #    model = pickle.load(f.read(), encoding='latin1')
        #with open(w2v_path, 'wb') as output:
        #    pickle.dump(model, output)
        model = gensim.models.Word2Vec.load(w2v_path)

    print("Done loading the Gensim model.")

    w2v_cos, w2v_cond = {}, {}
    # List of all the norms in the model. Used in normalization of cond prob.
    wordlist = set([])
    # Note that the cosine is the same as dot product for cbow vectors
    for cue in norms:
        if cue not in model:
            continue
        if cue not in w2v_cos:
            w2v_cos[cue], w2v_cond[cue] = {}, {}
        wordlist.add(cue)

        targetlist = list(set(list(norms[cue].keys()) + list(norms.keys())))

        for target in targetlist:#norms:
            if target not in model:
                continue
            if target not in w2v_cos:
                w2v_cos[target], w2v_cond[target] = {}, {}
            if target not in w2v_cos[cue]:
                w2v_cos[cue][target] = model.similarity(cue, target)
                w2v_cos[target][cue] = w2v_cos[cue][target]

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    for cue in w2v_cos.keys():
        cue_context = []
        for w in wordlist:
            cue_context.append(model.similarity(cue, w))
        # Using words in the word_list to normalize the prob
        p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
        for target in w2v_cos[cue]:
            w2v_cond[cue][target] = np.exp(w2v_cos[cue][target] - p_cue)

    with open(w2vcond_pickle, 'wb') as output:
        pickle.dump(w2v_cond, output)
    with open(w2vcos_pickle, 'wb') as output:
        pickle.dump(w2v_cos, output)

    print("cosine size %d norms size %d cond size %d" %
          (len(w2v_cos), len(norms), len(w2v_cond)))

    return w2v_cos, w2v_cond


def get_lda(lda_path, norms=None, vocab_path=None,
            lambda_path=None, gamma_path=None, mu_path=None):
    """ Get the cond prob for word representations learned by
    Hoffman-VBLDA-based code.
    Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
    p(topic|cue) = theta_cue[topic] because document is the cue
    vocab_path: the word2id mappings
    """
    if os.path.exists(lda_path):
        return load_scores(lda_path)

    word2id = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            word2id[line.split()[0]] = int(line.split()[1])

    # Getting the topic-word probs -- p(topic|cue)
    topics = np.loadtxt(lambda_path)
    num_topics = topics.shape[0]
    print("number of topics %d" % num_topics)
    print("lambda", topics.shape)

    # Normalize the topic-word probs
    if mu_path is None:
        for k in range(num_topics):
            topics[k] = topics[k] / sum(topics[k])
    else:
        mu = (np.loadtxt(mu_path))
        for k in range(num_topics):
            denom = topics[k] + mu[k]
            topics[k] = topics[k] / denom
            assert sum(topics[k] + (mu[k] / denom)) == len(word2id)

    gamma = np.loadtxt(gamma_path)
    print("gamma", gamma.shape)

    condprob = {}
    for cue in norms:
        if cue not in word2id.keys():
            continue
        if cue not in condprob:
            condprob[cue] = {}

        # Topic distribution of the document associated with cue
        cueid = word2id[cue]  # TODO would this be true
        cue_topics_dist = gamma[cueid] / sum(gamma[cueid])  # Normalize gamma
        # Calculate the cond prob for all the targets given the cue, and
        # also all the possible cues
        targetlist = list(set(list(norms[cue].keys()) + list(norms.keys())))
        for target in targetlist:
            if target not in word2id.keys():
                continue
            if target not in condprob[cue]:
                condprob[cue][target] = 0
            targetid = word2id[target]
            for k in range(num_topics):
                condprob[cue][target] += topics[k][targetid] *\
                    cue_topics_dist[k]

    with open(lda_path, 'wb') as output:
        pickle.dump(condprob, output)

    return condprob


def get_allpairs(allpairs_pickle, norms, cbow=None, sg=None, lda=None):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_pickle):
        return load_scores(allpairs_pickle)

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
            allpairs.append((cue, target))

    print("cues and targets in norms %d" % len(normpairs))
    print("cues and targets in norms and other data %d" % len(allpairs))

    with open(allpairs_pickle, 'wb') as output:
        pickle.dump(allpairs, output)
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
    assym_pairs = set()
    for cue, target in allpairs:
        if not (cue in norms and target in norms[cue]):
            continue
        if not (target in norms and cue in norms[target]):
            continue
        assym_pairs.add((min(cue, target), max(cue, target)))
    return list(assym_pairs)


def get_tuples(norms, allpairs):
    """ Find all the three words for which P(w2|w1), P(w3|w2),
    and P(w3|w1) exist.
    This is equivalent to the subset of the combination of 3-length ordered
    tuples that their pairs exist in Nelson norms.
    """
    # TODO make faster
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
    return tuples


# TODO need to rewrite
def read_gensimlda(ldapath, norm2doc_path, corpus_path, norms, word_list):
    """ Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
        p(topic|cue) = theta_cue[topic] because document is the cue
    """
    outname = ldapath + ".pickle"
    if os.path.exists(outname):
        return load_scores(outname)

    try:
        lda = gensim.models.LdaModel.load(ldapath)
    except:
        with open('/tmp/badfile', 'a') as f:
            f.write(ldapath + '\n')
            return None
    word2id ={}
    for wordid, word in lda.id2word.items():
        word2id[word] = wordid

    # Getting the topic-word probs
    topics =  lda.state.get_lambda()
    # Normalize the topic-word probs
    for k in range(lda.num_topics):
        topics[k] = topics[k]/topics[k].sum()

    # norm_id --> (doc_id, norm_freq)
    norm2doc = load_scores(norm2doc_path)
    corpus = gensim.corpora.MmCorpus(corpus_path)

    condprob = {}
    for cue in norms:
        if not (cue in word_list): continue
        if not cue in condprob:
            condprob[cue] = []
        # Topic distribution of the document associated with cue
        doc_id = norm2doc[cue.encode()][0]
        gamma, _ = lda.inference([corpus[doc_id]])
        doc_topics_dist = gamma[0] / sum(gamma[0])  # normalize distribution

        for target in norms[cue]:
            if not (target in word_list): continue
            targetid = word2id[target]
            #
            for k in range(lda.num_topics):
                condprob[cue][target] +=  topics[k][targetid] * doc_topics_dist[k]

    with open(outname, 'wb') as output:
        pickle.dump(condprob, output)

    return condprob



def read_filter_words(filename):
    """
    Read the subset of words from Nelson norms that is used in Griffiths et al (2007)
    """

    filters = sio.loadmat(filename)
    filters = filters['W'][0].tolist()
    words = []
    for f in filters:
        words.append(f.tolist()[0].lower())
    return words

def mm2dt(mmcorpus_path, vocab_path):
    """ Read a mmcorpus file and write it as a doc-term matrix """
    mmcorpus = gensim.corpora.MmCorpus(mmcorpus_path)
    id2word = gensim.corpora.Dictionary.load_from_text(vocab_path)

    norm2doc = load_scores(mmcorpus_path + ".norm2doc")
    n_vocab = len(id2word.keys())
    n_doc = len(norm2doc.keys())
    dt = np.zeros((n_doc, n_vocab), dtype=np.intc)
    #
    for i in range(len(list(mmcorpus))):
        for (w, count) in mmcorpus[i]:
            dt[i, w - 1] = count
    #
    vocab = np.zeros(n_vocab, dtype=object)
    for (wid, word) in id2word.items():
        vocab[wid] = word
    #
    return dt, vocab





