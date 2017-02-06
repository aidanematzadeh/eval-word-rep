import sys, pickle, os, numpy as np, json
sys.path.append('/home/stephan/python/gibbs_lda')
import gibbs_lda


inputFolder = '/shared_hd0/corpora/TASA/tsg_tasa'
V = np.load(open(os.path.join(inputFolder,'5w_V'), 'rb'))
D = np.load(open(os.path.join(inputFolder,'5w_D'), 'rb'))

#load the vocab
with open(os.path.join(inputFolder,'5w_id2word.json')) as data_file:    
    vocab_json = json.load(data_file)

vocab = dict(zip([int(x) for x in vocab_json.keys()], vocab_json.values()))
print vocab.values()[1:10]



#alpha is doc-> topic smoothing, beta is topic -> word
params = {'lda_k':300, 'lda_beta': 0.01, 'lda_alpha': .16}

#what were the reuslts in the paper

reload(gibbs_lda)
tasa_lda = gibbs_lda.LDA(V, D, vocab, params)

print('Burning in')
tasa_lda.burnIn(800, thinning=10)
	
print('Sampling')
tasa_lda.sample(240, thinning=10)


#this is either numTokens * numTopics or numTokens * numSamples
#with 28m tokens, this gets crushingly large . 56m integers

pickle.dump(tasa_lda, open( "tasa_skipgram_lda.pkl", "wb" ))

#!!! did not overwrite... why?