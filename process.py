import os
import gensim
import numpy as np
import scipy.io as sio
import scipy
import pickle
from collections import defaultdict
import os.path

# Reads and process data used in the evaluation.

def load_scores(path):
    """loads a pickle file.
    """
    with open(path, 'rb') as f:
        scores = pickle.load(f)
    return scores

def defaultdict_float():
    return defaultdict(float)

def get_norms(picklefilename, norms_path=None):
    """ Read Nelson norms for the evaluation methods.
    If a pickle file exists, load and return the norms.
    Otherwise, read the norms from the dir and write a pickle file to picklefilename.
    Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
    """
    if os.path.exists(picklefilename):
        return load_scores(picklefilename)

    # The value of zero means that the norms[cue][target] does not exist.
    norms = defaultdict(defaultdict_float)
    for filename in os.listdir(norms_path):
        norm_file = open(norms_path + "/" + filename, 'r', encoding = "ISO-8859-1")
        norm_file.readline()
        for line in norm_file:
            nodes = line.strip().split(',')
            cue = nodes[0].strip().lower()
            target = nodes[1].strip().lower()
            # Check if the cue, target pair is normed that is p(cue|target) also exists
            # The next line was used in NIPS submission. Commented in Oct 2016.
            #if nodes[2].strip() == "YES":
            norms[cue][target] = float(nodes[5]) #FSG, p(target|cue)
    #
    with open(picklefilename ,'wb') as output:
        pickle.dump(norms, output)

    return norms

def get_cbow(cbow_cos_path, cbow_cond_path, norms=None, cbow_binary=None):
    """ Read Google CBOW (cbow) representations for words in norms and popluate
    their similarities using cosine and p(w2|w1).
    Uses gensim to load the representations.
    """
    if os.path.exists(cbow_cos_path):#.endswith("pickle"):
        cbow_cos = load_scores(cbow_cos_path)
        cbow_cond = load_scores(cbow_cond_path)
        return cbow_cos, cbow_cond

    model = gensim.models.Word2Vec.load_word2vec_format(cbow_binary, binary=True)
    print("Done loading the Gensim model.")

    cbow_cos = defaultdict(defaultdict_float)
    cbow_cond = defaultdict(defaultdict_float)
    # list of all the norms that have a CBOW rep. Used in normalization of cond prob.
    wordlist = set([])
    # Note that the cosine is the same as dot product for cbow vectors
    for cue in norms:
        if not cue in model: continue
        wordlist.add(cue)
        for target in norms[cue]:
            if not target in model: continue
            wordlist.add(target)
            if cbow_cos[cue][target] == 0:
                cbow_cos[cue][target] = model.similarity(cue, target)
                cbow_cos[target][cue] = cbow_cos[cue][target]

    # Calculate p(target|cue) where cue is w1 and target is w2
    # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
    for cue in cbow_cos.keys():
        cue_context = []
        for w in wordlist:
            cue_context.append(model.similarity(cue, w))
        # Using words in the word_list to normalize the prob
        p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
        for target in cbow_cos[cue]:
            cbow_cond[cue][target] = np.exp(cbow_cos[cue][target] - p_cue)

    with open(cbow_cond_path , 'wb') as output:
        pickle.dump(cbow_cond, output)
    with open(cbow_cos_path, 'wb') as output:
        pickle.dump(cbow_cos, output)

    print("cosine size %d norms size %d cond size %d", (len(cbow_cos), len(norms), len(cbow_cond)))

    return cbow_cos, cbow_cond

def get_lda(lda_path, norms=None, vocab_path=None, lambda_path=None, gamma_path=None):
    """ Get the cond prob for word representations learned using Hoffman-VBLDA-based code.
        Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
        p(topic|cue) = theta_cue[topic] because document is the cue
        vocab_path: the word2id mappings
    """
    if os.path.exists(lda_path):
        return load_scores(lda_path)

    word2id, id2word = {}, {}
    with open(vocab_path, 'r') as f:
        for line in f:
            word2id[line.split()[0]] = int(line.split()[1])
            id2word[int(line.split()[1])] = line.split()[0]

    # Getting the topic-word probs -- p(topic|cue)
    topics = (np.loadtxt(lambda_path)).T
    print("lambda", topics.shape)
    num_topics = topics.shape[0]
    print("number of topics %d" % num_topics)
    # Normalize the topic-word probs
    for k in range(len(topics)):
        topicsk = list(topics[k, :])
        topics[k] = topicsk / sum(topicsk)

    print("topics", topics)
    #
    gamma = np.loadtxt(gamma_path)
    print("gamma", gamma.shape)
    print("Gamma", gamma)
    #
    condprob = defaultdict(defaultdict_float)
    import math #TODO
    for cue in norms:
        if not (cue in word2id.keys()): continue
        # Topic distribution of the document associated with cue
        cueid = word2id[cue] #TODO would this be true
        cue_topics_dist = gamma[cueid] / sum(gamma[cueid])  # normalize distribution
        print(cueid, sum(gamma[cueid]),sum(gamma[cueid,:]))
        for target in norms[cue]:
            if not (target in word2id.keys()): continue
            targetid = word2id[target]
            #
            for k in range(num_topics):
                condprob[cue][target] +=  topics[k][targetid] * cue_topics_dist[k]

            #TODO
            if math.isnan(condprob[cue][target]):
                print(cue, target, condprob[cue][target])

    with open(lda_path, 'wb') as output:
        pickle.dump(condprob, output)

    return condprob

def get_allpairs(allpairs_path, norms, cbow=None, lda=None):
    """ Get all cue-target pairs that occur in all of our evaluation sets, that is,
    Nelson norms, cbow, and LDA.
    """
    if os.path.exists(allpairs_path):#.endswith("pickle"):
        return load_scores(allpairs_path)

    allpairs = []
    normpairs = []
    for cue in norms:
        for target in norms[cue]:
            normpairs.append((cue, target))
            if cbow != None and cbow[cue][target] == 0: continue
            if lda != None and lda[cue][target] == 0: continue
            allpairs.append((cue, target))

    print("cues and targets in norms %d" % len(normpairs))
    print("cues and targets in norms and other data %d" % len(allpairs))

    #if not filter_path == None:
    #    self.assoc_list = self.read_filter_words(filter_path)
    #    print("assoc list from Griffiths et al", len(self.assoc_list))
    with open(allpairs_path, 'wb') as output:
        pickle.dump(allpairs, output)
    return allpairs

#TODO need to rewrite
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
    word2id =  {}
    for wordid, word in lda.id2word.items():
        word2id[word] = wordid

    # Getting the topic-word probs
    topics =  lda.state.get_lambda()
    # Normalize the topic-word probs
    for k in range(lda.num_topics):
        topics[k] = topics[k]/topics[k].sum()

    #norm_id --> (doc_id, norm_freq)
    norm2doc = load_scores(norm2doc_path)
    corpus = gensim.corpora.MmCorpus(corpus_path)

    #wordprob = np.zeros(lda.num_terms)
    #for i in range(lda.num_terms):
    #    wordprob[i] = np.sum(topics[:,i]) # this should be multiplied with the topic prob

    condprob = defaultdict(lambda: defaultdict(float))
    for cue in norms:
        if not (cue in word_list): continue
        #cueid = word2id[cue]
        # Topic distribution of the document associated with cue
        doc_id = norm2doc[cue.encode()][0]
        gamma, _ = lda.inference([corpus[doc_id]])
        doc_topics_dist = gamma[0] / sum(gamma[0])  # normalize distribution
        #doc_topics = lda.get_document_topics(corpus[doc_id], minimum_probability=0)

        for target in norms[cue]:
            if not (target in word_list): continue
            targetid = word2id[target]
            #
            for k in range(lda.num_topics):
                condprob[cue][target] +=  topics[k][targetid] * doc_topics_dist[k]

    with open(outname, 'wb') as output:
        pickle.dump(condprob, output)

    return condprob

def get_pair_scores(scores, ctpairs):
    """ Return the cue-target scores
    """
    pair_scores = []
    for cue, target in ctpairs:
        pair_scores.append(scores[cue][target])
    return pair_scores

def get_asym_pairs(norms, allpairs):
    """ Return the pairs for which both p(target|cue) and P(cue|target) exist.
    """
    assym_pairs = set()
    for cue, target in allpairs:
        if norms[cue][target] !=0 and norms[target][cue] !=0:
            assym_pairs.add((min(cue,target), max(cue,target)))
    return list(assym_pairs)

def get_tuples(norms, allpairs):
    """ Find all the three words for which P(w2|w1), P(w3|w2), and P(w3|w1) exist
    This is equivalent to the subset of the combination of 3-length ordered tuples
    that their pairs exist in Nelson norms.
    """
    tuples = []
    for w1 in norms:
        for w2 in norms[w1]:
            if not (w1, w2) in allpairs: continue
            if not w2 in norms: continue
            for w3 in norms[w2]:
                if not (w2, w3) in allpairs: continue
                if (w1, w3) in allpairs:
                    tuples.append((w1, w2, w3))
    return tuples


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





