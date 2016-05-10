#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

If you have the `pattern` package installed, this module will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

See scripts/process_wiki.py for a canned (example) script based on this
module.

Aida: Added a map-reduce function to collect the context of words instead of documents

"""


import gc
import bz2
import logging
import re
from xml.etree.cElementTree import iterparse  # LXML isn't faster, so let's go with the built-in solution
import multiprocessing
import os
import time
import pickle
from collections import defaultdict
#import numpy
from stop_words import get_stop_words
from gensim import utils

# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger('gensim.corpora.wikicorpus')

# ignore articles shorter than ARTICLE_MIN_WORDS characters (after full preprocessing)
ARTICLE_MIN_WORDS = 50


RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)  # comments
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)  # footnotes
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE)  # links to languages
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)  # remove URL, keep description
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE)  # simplify links, keep description
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of images
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of files
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)  # outside links
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)  # math content
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE)  # all other tags
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE)  # table formatting
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE)  # table cell formatting
RE_P14 = re.compile('\[\[Category:[^][]*\]\]', re.UNICODE)  # categories
# Remove File and Image template
RE_P15 = re.compile('\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)

# MediaWiki namespaces (https://www.mediawiki.org/wiki/Manual:Namespace) that
# ought to be ignored
IGNORED_NAMESPACES = ['Wikipedia', 'Category', 'File', 'Portal', 'Template',
                      'MediaWiki', 'User', 'Help', 'Book', 'Draft',
                      'WikiProject', 'Special', 'Talk']

stoplist = set(get_stop_words('en'))

def filter_wiki(raw):
    """
    Filter out wiki mark-up from `raw`, leaving only text. `raw` is either unicode
    or utf-8 encoded string.
    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text)


def remove_markup(text):
    text = re.sub(RE_P2, "", text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, "", text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, "", text)  # remove outside links
        text = re.sub(RE_P10, "", text)  # remove math content
        text = re.sub(RE_P11, "", text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description
        text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        if old == text or iters > 2:  # stop if nothing changed between two iterations or after a fixed number of iterations
            break

    # the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
    # TODO is this really desirable?
    text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text
    return text


def remove_template(s):
    """Remove template wikimedia markup.

    Return a copy of `s` with all the wikimedia markup template removed. See
    http://meta.wikimedia.org/wiki/Help:Template for wikimedia templates
    details.

    Note: Since template can be nested, it is difficult remove them using
    regular expresssions.
    """

    # Find the start and end position of each template by finding the opening
    # '{{' and closing '}}'
    n_open, n_close = 0, 0
    starts, ends = [], []
    in_template = False
    prev_c = None
    for i, c in enumerate(iter(s)):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = 0, 0
        prev_c = c

    # Remove all the templates
    s = ''.join([s[end + 1:start] for start, end in
                 zip(starts + [None], [-1] + ends)])

    return s


def remove_file(s):
    """Remove the 'File:' and 'Image:' markup, keeping the file caption.

    Return a copy of `s` with all the 'File:' and 'Image:' markup replaced by
    their corresponding captions. See http://www.mediawiki.org/wiki/Help:Images
    for the markup details.
    """
    # The regex RE_P15 match a File: or Image: markup
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s


def tokenize(content):
    """
    Tokenize a piece of text from wikipedia. The input string `content` is assumed
    to be mark-up free (see `filter_wiki()`).

    Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
    that 15 characters (not bytes!).
    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
            if 2 <= len(token) <= 15 and not token.startswith('_') and not token in stoplist]


def get_namespace(tag):
    """Returns the namespace of tag."""
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace"
                         % namespace)
    return namespace
_get_namespace = get_namespace


def extract_pages(f, filter_namespaces=False):
    """
    Extract pages from a MediaWiki database dump = open file-like object `f`.

    Return an iterable over (str, str, str) which generates (title, content, pageid) triplets.

    """
    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    pageid_path = "./{%(ns)s}id" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            if any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                continue

            text = elem.find(text_path).text

            ns = elem.find(ns_path).text
            if filter_namespaces and ns not in filter_namespaces:
                text = None

            pageid = elem.find(pageid_path).text
            yield title, text or "", pageid     # empty page will yield None

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()
_extract_pages = extract_pages  # for backward compatibility


def process_article(args):
    """
    Parse a wikipedia article, returning its content as a list of tokens
    (utf8-encoded strings).
    """
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text)
    return result, title, pageid

def defaultdict_int():
    return defaultdict(int)

contexts_clear = False
contexts_local = defaultdict(defaultdict_int)

class GetWordContextWrapper(object):
    def __init__(self, hwsize, wordlist):
        self.hwsize = hwsize
        self.wordlist = wordlist
    def __call__(self, args):
        get_word_context(args, self.hwsize, self.wordlist)

def get_word_context(args, hwsize, wordlist):
    """
    Parse a wikipedia article, return the context of words in word_list as a dictionary
    """
    global contexts_clear, contexts_local
    if contexts_clear:
        contexts_local.clear()
        contexts_local = defaultdict(defaultdict_int)
        gc.collect()
        contexts_clear = False
        assert(len(contexts_local) == 0)
        print('%d context cleared' % os.getpid())

    text, lemmatize, title, pageid  = args
    if any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
        return

    text = filter_wiki(text)
    result = []
    for i in range(hwsize):
        result.append(b't-start')

    if lemmatize:
        result += utils.lemmatize(text)
    else:
        result += tokenize(text)

    for i in range(hwsize):
        result.append(b"t-end")

    for index in range(len(result)):
        word = result[index]
        if not word.decode() in wordlist:
            continue
        for cword in result[index-hwsize:index+hwsize+1]:
            contexts_local[word][cword] += 1 #.append(' '.join(result[index-hwsize:index+hwsize+1]))
    # TODO

def fetch_contexts(x):
    global contexts_clear, contexts_local
    contexts_clear = True
    print('%d fetching context' % os.getpid())
    return os.getpid(), contexts_local


class WikiCorpus(TextCorpus):
    """
    Treat a wikipedia articles dump (\*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fname, wordlist, wsize, norm2docfile, wikiflag=True,  processes=None, lemmatize=utils.has_pattern(), dictionary=None, filter_namespaces=('0',)):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.

        If `pattern` package is installed, use fancier shallow parsing to get
        token lemmas. Otherwise, use simple regexp tokenization. You can override
        this automatic logic by forcing the `lemmatize` parameter explicitly.

        Aida: If getcontext is true, each document would consist of the context of a word
        wordlist: words for which we extract the context
        wsize: window size to look for the context

        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 2)
        self.processes = processes
        self.lemmatize = lemmatize

        #
        self.wsize = wsize
        self.wordlist = wordlist
        self.norm2docfile = norm2docfile
        self.contexts = defaultdict(lambda: defaultdict(int))
        # This says if we are processing wiki or a text file
        self.wikiflag = wikiflag
        #
        self.word2id = {}
        self.id2word = {}
        self.word_id_free = 1

        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

    def get_texts(self):
        """
        Iterate over the dump, returning context of each word as a document
        Only collects the context for words in wordlist.
        Map function collect the context in a given document
        Reduce function put the contexts of a word together

        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        """
        if len(self.contexts) == 0:
            logger.info("Calling get_all_contexts()")
            self.get_all_contexts()

        word2doc = {}
        for docid, (word_id, d) in enumerate(self.contexts.items()):
            word2doc[self.id2word[word_id]] = [docid, d[word_id]]
            yield (self.id2word[cword_id]
                    for cword_id, count in d.items() for i in range(count))
            # This would not keep the nelson norms in the document
            #yield (self.id2word[cword_id]
            #        for cword_id, count in d.items() for i in range(count) if cword_id != word_id)


        with open(self.norm2docfile, 'wb') as f:
            pickle.dump(word2doc, f)


    def get_all_contexts(self):
        def word_id(word):
            if word not in self.word2id:
                self.word2id[word] = self.word_id_free
                self.id2word[self.word_id_free] = word
                self.word_id_free += 1
            return self.word2id[word]

        def merge_contexts():
            pid_set = set()
            for pid, contexts_local in pool.imap(fetch_contexts, [1] * self.processes):
                assert(pid not in pid_set)
                pid_set.add(pid)
                for word, d in contexts_local.items():
                    for cword, count in d.items():
                        self.contexts[word_id(word)][word_id(cword)] += count
                gc.collect()

        hwsize = self.wsize//2
        wordlist = self.wordlist
        #
        if self.wikiflag:
            texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        else:
            tfile = open(self.fname, 'r')
            texts = ((line, self.lemmatize, "title", "pageid") for line in tfile)
            self.processess = 1 #TODO
        #
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        index = 0
        chunk_size = 25 * self.processes
        fetch_frequency = 2000000 - 2000000 % chunk_size
        for group in utils.chunkize(texts, chunksize=chunk_size, maxsize=1):
            logger.info("processing %d %d %f" % (index, len(group), time.time()))
            pool.map(GetWordContextWrapper(hwsize, wordlist), group)#, chunksize=500)
            index += len(group)
            # combining the contexts
            if index % fetch_frequency == 0:
                merge_contexts()
          #  if index >= 100000: break
        #
        merge_contexts()
        pool.terminate()


# endclass WikiCorpus
