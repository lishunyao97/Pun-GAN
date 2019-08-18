# encoding: utf-8
from nltk.corpus.reader.wordnet import WordNetCorpusReader

wn = WordNetCorpusReader(YOUR_WORDNET_PATH, '.*')  # 这种方式就会有函数补全
print('wordnet version %s: %s' % (wn.get_version(), YOUR_WORDNET_PATH))

print'get gloss from sensekey......'
key = 'dance%1:04:00::'
lemma = wn.lemma_from_key(key)
synset = lemma.synset()
print synset.definition()
