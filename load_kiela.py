import gensim
import sys
sys.path.append('/home/stephan/python/fancysg')
from sg import SkipGram
test = SkipGram.load('/shared_hd1/word-representations/data/kiela/simrel/300-joint-usfnorms-all.model')

import pdb
pdb.set_trace()
