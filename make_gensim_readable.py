#!/usr/local/bin/python

import argparse
import numpy

parser = argparse.ArgumentParser(description='Convert a retrofit vector file into a format readable with gensim keyed_text')
# takes an input_path as sole argument
parser.add_argument('--input_path', type=str, help='path to the model to convert')
args = parser.parse_args()

def read_word_vecs(filename, normalize=False):
    ''' Read all the word vectors and (optionally) normalize them.
      This is a modified version of the f in retrofit_original.py.
    '''
    print('\nReading the vectors in '+filename+'\n')
    wordVectors = {}
    fileObject = open(filename, 'r')

    for line in fileObject:
        line = line.strip().lower()
        word = line.split()[0]
        wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
        for index, vecVal in enumerate(line.split()[1:]):
          wordVectors[word][index] = float(vecVal)
        if normalize:
            wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    return wordVectors

def output_word_vecs(wordVectors, outFileName, precision=4):
    print('\nWriting down the vectors in '+outFileName+'\n')
    outFile = open(outFileName, 'w')
    # write an initial line with the size of the vocabulary and the dimensionality
    outFile.write(str(len(wordVectors)) + ' ' + str(len(wordVectors[wordVectors.keys()[0]])) +'\n')  
    for word, values in wordVectors.iteritems():
        outFile.write(word+' ')
        for val in wordVectors[word]:
        #outFile.write('%.4f' %(val)+' ')
            outFile.write('%.*f' % (precision, val) + ' ')
        outFile.write('\n')
    outFile.close()    

def make_gensim_readable(input_path, normalize=False, precision=4):      
    wordVectors = read_word_vecs(input_path, normalize)
    outFileName = input_path+'_converted'
    output_word_vecs(wordVectors, outFileName, precision)

if __name__ == "__main__":
    make_gensim_readable(args.input_path, normalize=False, precision=4)