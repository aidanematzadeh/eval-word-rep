import os
import gensim
import numpy as np
import scipy.io as sio
import scipy
import pickle


class ProcessData:
    """ This class reads and process data used for evaluation.
    """
    def __init__(self, filter_path=None, google_path=None, nelson_path=None, readflag=True):
        self.word_list = set()
        if not readflag:
            self._calc_scores(filter_path, google_path, nelson_path)

    def load_scores(self, path):
        with open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def _calc_scores(self, filter_path, google_path, nelson_path):
        # Read the word list that the evaluation is on. These words should occur in
        # both Nelson norms and Google vectors

        assoc_list = self.read_filter_words(filter_path)
        print("assoc list from Griffiths et al", len(assoc_list))

        word2vec_model = self.load_word2vec_model(google_path)
        # words in common between norms and word2vec
        for word in assoc_list:
            if word in word2vec_model:
                self.word_list.add(word)
        print("words from assoc list that occuer in word2vec vectors", len(self.word_list))

        norms_fsg = self.read_norms(nelson_path, self.word_list)
        print("norm list", len(norms_fsg))
        #
        with open('norms.pickle' ,'wb') as output:
            pickle.dump(norms_fsg, output)
        #
        word2vec_cos, word2vec_cond = self.read_word2vec(word2vec_model, norms_fsg, self.word_list)
        with open('word2vec_cond.pickle' , 'wb') as output:
            pickle.dump(word2vec_cond, output)
        #
        with open('word2vec_cos.pickle', 'wb') as output:
            pickle.dump(word2vec_cos, output)
        print("size", len(word2vec_cos), len(norms_fsg), len(word2vec_cond))


    def load_word2vec_model(self, google_path):
        model = gensim.models.Word2Vec.load_word2vec_format(google_path, binary=True)
        print("Done loading the Gensim model.")
        return model

    def read_word2vec(self, model, norms, word_list):
        """ Read Word2Vec representations for words in norms and popluate
        their similarities using cosine and p(w2|w1).

        We assume that all words in word_list have a word2vec representation.
        """
        word2vec_cos = {}
        word2vec_logcond = {}
        for cue in norms:
            word2vec_cos[cue] = {}
            word2vec_logcond[cue] = {}
        #
        # Note that the cosine is the same as dot product for word2vec vectors
        for cue in norms:
            for target in norms[cue]:
                try:
                    word2vec_cos[cue][target] = model.similarity(cue, target)
                except KeyError:
                    print(cue, target)

        #cue w1, target w2, calculating p(w2|w1)
        # log(p(w2|w1)) = log(exp(w2.w1)) - log(sum(exp(w',w1)))
        for cue in norms:
            cue_context = []
            for w in word_list:
                cue_context.append(model.similarity(cue, w))
            # Using words that occuerd as norms/targets with the cue
            #p_cue = scipy.misc.logsumexp(np.asarray(list(word2vec_cos[cue].values())))
            #
            # Using words in the word_list to normalize the prob
            p_cue = scipy.misc.logsumexp(np.asarray(cue_context))
            for target in norms[cue]:
                word2vec_logcond[cue][target] = np.exp(word2vec_cos[cue][target] - p_cue)

        return word2vec_cos, word2vec_logcond


    def read_norms(self, norms_dir, word_list):
        """ Read Nelson norms for the evaluation methods.
        Limit the words to the one that are in the word_list
        Norms are formatted as: CUE, TARGET, NORMED?, #G, #P, FSG, BSG,
        """
        norms = {}
        for filename in os.listdir(norms_dir):
            norm_file = open(norms_dir + "/" + filename, 'r', encoding = "ISO-8859-1")
            norm_file.readline()
            for line in norm_file:
                nodes = line.strip().split(',')
                cue = nodes[0].strip().lower()
                target = nodes[1].strip().lower()

                if len(word_list) > 0 and \
                        not (cue in word_list and target in word_list): continue

                if not cue in norms:
                    norms[cue] = {}

                norms[cue][target] = float(nodes[5]) #FSG

#        for cue in norms:
#            for target in norms[cue]:
#                if target in norms.keys() and not (cue in norms[target]):
#                    norms[target][cue] = 0

        return norms

    def get_pairs(self, norms):
        pairs = set()
        for cue in norms:
            for target in norms[cue]:
                if target in norms and cue in norms[target]:
                    pairs.add((min(cue,target), max(cue,target)))
        return list(pairs)

    def get_tuples(self, norms):
        """ Find all the three words for which P(w2|w1), P(w3|w2), and P(w3|w1) exist
        This is equivalent to the subset of the combination of 3-length ordered tuples
        that their pairs exist in Nelson norms.
        """
        tuples = []

        for w1 in norms:
            for w2 in norms[w1]:
                if not w2 in norms:
                    continue

                for w3 in norms[w2]:
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







