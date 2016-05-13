import os
import gensim
import numpy as np
import scipy.io as sio
import scipy
import pickle
from collections import defaultdict

def defaultdict_float():
    return defaultdict(float)

def read_norms(norms_dir):
    """ Read Nelson norms for the evaluation methods.
    Limit the words to the one that are in the word_list
    Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
    """
    # The value of zero means that the norms[cue][target] does not exist.
    norms = defaultdict(defaultdict_float)
    for filename in os.listdir(norms_dir):
        norm_file = open(norms_dir + "/" + filename, 'r', encoding = "ISO-8859-1")
        norm_file.readline()
        for line in norm_file:
            nodes = line.strip().split(',')
            cue = nodes[0].strip().lower()
            target = nodes[1].strip().lower()
            # Check if the cue, target pair is normed that is p(cue|target) also exists
            if nodes[2].strip() == "YES":
                norms[cue][target] = float(nodes[5]) #FSG, p(target|cue)
            #elif nodes[2].strip() != "NO":
            #    print("Normed is not yes, is not no, what?", nodes[2])
    return norms

class ProcessData:
    """ This class reads and process data used for evaluation.
    """
    def __init__(self,  nelson_path, word2vec_path, wikiwords_path, outdir, commonwords):
        self.common_words = set()
        # The path where the pickle files are written
        self.outdir = outdir + "/"

        # Nelson Norms
        if nelson_path.endswith("pickle"):
            self.norms_fsg = self.load_scores(nelson_path)
        else:
            self.norms_fsg = self.read_norms(nelson_path)

        #common words
        if commonwords.endswith("pickle"):
            self.common_words = self.load_scores(commonwords)
        else:
            self.common_words = self.get_words(self.norms_fsg, word2vec_path, wikiwords_path)

        # Word2vec
        if word2vec_path.endswith("pickle"):
            self.word2vec_cond = self.load_scores(word2vec_path)
            self.word2vec_cos = self.load_scores(word2vec_path.replace('cond','cos'))
        else:
            word2vec_model = self.load_word2vec_model(word2vec_path)
            print("common words %d" % len(self.common_words))
            self.word2vec_cos, self.word2vec_cond = self.read_word2vec(word2vec_model, self.norms_fsg, self.common_words)

    def load_scores(self, path):
        with open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def load_word2vec_model(self, google_path):
        model = gensim.models.Word2Vec.load_word2vec_format(google_path, binary=True)
        print("Done loading the Gensim model.")
        return model

    def read_word2vec(self, model, norms, word_list):
        """ Read Word2Vec representations for words in norms and popluate
        their similarities using cosine and p(w2|w1).

        We assume that all words in word_list have a word2vec representation.
        """
        word2vec_cos = defaultdict(defaultdict_float)
        word2vec_cond = defaultdict(defaultdict_float)
        # Note that the cosine is the same as dot product for word2vec vectors
        for cue in norms:
            if not (cue in word_list): continue
            for target in norms[cue]:
                if not (target in word_list): continue
                word2vec_cos[cue][target] = model.similarity(cue, target)

        # Calculate p(target|cue) where cue is w1 and target is w2
        # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
        for cue in norms:
            if not cue in word_list: continue
            cue_context = []
            #TODO
            for w in word_list:
                cue_context.append(model.similarity(cue, w))
            # Using words that occuerd as norms/targets with the cue
            #p_cue = scipy.misc.logsumexp(np.asarray(list(word2vec_cos[cue].values())))
            #
            # Using words in the word_list to normalize the prob
            p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
            for target in norms[cue]:
                word2vec_cond[cue][target] = np.exp(word2vec_cos[cue][target] - p_cue)


        with open(self.outdir + 'word2vec_cond.pickle' , 'wb') as output:
            pickle.dump(word2vec_cond, output)
        #
        with open(self.outdir + 'word2vec_cos.pickle', 'wb') as output:
            pickle.dump(word2vec_cos, output)
        print("size", len(word2vec_cos), len(norms), len(word2vec_cond))

        return word2vec_cos, word2vec_cond

    def read_lda(self, ldapath, norm2doc_path, corpus_path, norms, word_list):
        """ Calculate p(target|cue) = sum_topics{p(target|topic) p(topic|cue)}
            p(topic|cue) = theta_cue[topic] because document is the cue
        """

        lda = gensim.models.LdaModel.load(ldapath)
        word2id =  {}
        for wordid, word in lda.id2word.items():
            word2id[word] = wordid

        # Getting the topic-word probs
        topics =  lda.state.get_lambda()
        #Normalize the topic-word probs
        for k in range(lda.num_topics):
            topics[k] = topics[k]/topics[k].sum()

        #norm_id --> (doc_id, norm_freq)
        norm2doc = self.load_scores(norm2doc_path)
        corpus = gensim.corpora.MmCorpus(corpus_path)

        #wordprob = np.zeros(lda.num_terms)
        #for i in range(lda.num_terms):
        #    wordprob[i] = np.sum(topics[:,i]) # this should be multiplied with the topic prob

        condprob = defaultdict(defaultdict_float)
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

        return condprob

    def read_norms(self, norms_dir):
        """ Read Nelson norms for the evaluation methods.
        Limit the words to the one that are in the word_list
        Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
        """
        norms = read_norms(norms_dir)
        with open(self.outdir + 'norms.pickle' ,'wb') as output:
            pickle.dump(norms, output)

        return norms

    def get_words(self, norms, word2vec_path, lda_words_path):
        """ Get the word list that the evaluation is on. These words should occur in
        both Nelson norms, word2vec, and LDA.
        """
        words = set([])
        remove_words = []
        for cue in norms:
            words.add(cue)
            for target in norms[cue]:
                words.add(target)
        print("cues and targets in norms %d" % len(words))

        word2vec_model = self.load_word2vec_model(word2vec_path)
        for word in words:
            if not word in word2vec_model:
                remove_words.append(word)

        for word in remove_words:
            words.remove(word)
        remove_words = []
        print("cues and targets in norms and word2vec %d" % len(words))

        lda_file = open(lda_words_path, 'r')
        lda_words = set([line.split()[1] for line in lda_file])
        print("size of lda words %d" % len(lda_words))

        for word in words:
            if not word in lda_words:
                remove_words.append(word)
        for word in remove_words:
            words.remove(word)
        remove_words = []

        print("cues and targets in norms, word2vec, and LDA %d" % len(words))
        #if not filter_path == None:
        #    self.assoc_list = self.read_filter_words(filter_path)
        #    print("assoc list from Griffiths et al", len(self.assoc_list))
        with open(self.outdir + 'common_words.pickle', 'wb') as output:
            pickle.dump(words, output)
        return words


    def get_ct_scores(self, scores, ctpairs):
        """ Return the cue-target scores
        """
        probs = []
        for cue, target in ctpairs:
            probs.append(scores[cue][target])
        return probs



    def get_ct_pairs(self, norms, word_list):
        """ Return the cue-target pairs that exist in all models.
        """
        pairs = set()
        for cue in norms:
            if not cue in word_list: continue
            for target in norms[cue]:
                if not target in word_list: continue
                pairs.add((cue, target))
        return list(pairs)





    def get_pairs(self, norms, word_list):
        """ Return the pairs for which both p(target|cue) and P(cue|target)
        exist.
        """
        pairs = set()
        for cue in norms:
            if not cue in word_list: continue
            for target in norms[cue]:
                if not target in word_list: continue
                #
                if target in norms:
                    if cue in norms[target]:
                        pairs.add((min(cue,target), max(cue,target)))
        return list(pairs)



    def get_tuples(self, norms, word_list):
        """ Find all the three words for which P(w2|w1), P(w3|w2), and P(w3|w1) exist
        This is equivalent to the subset of the combination of 3-length ordered tuples
        that their pairs exist in Nelson norms.
        """
        tuples = []
        for w1 in norms:
            if not w1 in word_list: continue
            for w2 in norms[w1]:
                if not w2 in word_list: continue
                if not w2 in norms: continue

                for w3 in norms[w2]:
                    if not w3 in word_list: continue
                    if w3 in norms[w1]:
                        tuples.append((w1, w2, w3))
        return tuples


    def read_filter_words(self, filename):
        """
        Read the subset of words from Nelson norms that is used in Griffiths et al (2007)
        """

        filters = sio.loadmat(filename)
        filters = filters['W'][0].tolist()
        words = []
        for f in filters:
            words.append(f.tolist()[0].lower())

        return words







