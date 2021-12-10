# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:59:55 2021

@author: Lucas Puddifoot, Jort Gutter, Damy Hillen
"""

import pyserini
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
import numpy as np
import faiss
import json


index_reader = IndexReader("./indexes/wapo")
searcher = SimpleSearcher('./indexes/wapo')

# document we use is 95195e84-32fe-11e1-825f-dabc29fd7071
doc = searcher.doc('95195e84-32fe-11e1-825f-dabc29fd7071')
raw = doc.raw()
content = json.loads(raw)
text = "\n".join([line['content'] for line in content['contents'] if line['type'] in ['sanitized_html', 'title']])

# all of these can be retrieved using the index reader
N = index_reader.stats()['documents']
tf = index_reader.get_document_vector('95195e84-32fe-11e1-825f-dabc29fd7071')
df = {term: (index_reader.get_term_counts(term, analyzer=None))[0] for term in tf.keys()}

# building the idf from these components
# idf = log((1 + N) / number of documents where the term appears)
idf = {term: np.log((N + 1)/df[term]) for term in tf.keys()}
tf_idf = {term: tf[term]*idf[term] for term in tf.keys()}

top_100 = np.array(sorted([[idf, term] for term, idf in tf_idf.items()], reverse=True)[:100])
# query builder
# see if the query builder thing with the score might improve getting documents, but probably wont because they are weighted later
# using edge strength for re-weighting might be better but that is for after the graph is made
query = ' '.join(top_100[:,1])
print(query)

# test search with the query?
print("resuts")

hits = searcher.search(query)
for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
    # print(hits[i].raw)

# manually remove the document you are searching for because this is done in anserini
# baseline is running this write the topic then use trec eval. alos use rm3 as well as bm25 for the baseline
# searcher.set_rm3() for rm3 reweighting
# rm3 does ~~ bm25 -> top 10 doc look at term distribution -> depending on prob of words occuring added to query and weighted
# compare
