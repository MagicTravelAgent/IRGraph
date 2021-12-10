# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:59:55 2021

@author: Lucas Puddifoot, Jort Gutter, Damy Hillen
"""

from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from document import Document
from tqdm import tqdm
import re


def make_topic_dict():
    with open('./background_linking.txt') as f:
        topics = f.readlines()

    nums = []
    ids = []

    for x in topics:
        num_search = re.search('<num> Number: (.*) </num>', x, re.IGNORECASE)
        id_search = re.search('<docid>(.*)</docid>', x, re.IGNORECASE)
        if num_search:
            num = num_search.group(1)
            nums.append(num)
        if id_search:
            id = id_search.group(1)
            ids.append(id)

    topic_dict = {nums[i]: ids[i] for i in range(len(nums))}
    return topic_dict

# =========================== Pipeline ===========================

# make the topic dictionary with keys being topic number and value being document id
topics = make_topic_dict()

# create the searcher for the index
searcher = SimpleSearcher('./indexes/wapo')
searcher.set_bm25()
# create index reader
index_reader = IndexReader("./indexes/wapo")

tokenized_texts = {}

# open the output file
out_file = open("topic_rels.txt", "w")

# loop through the topic numbers
for topic_number in tqdm(topics):
    # getting the document ID for the topic number
    document_id = topics[topic_number]

    # making a query for this document:
    doc = Document(simple_searcher=searcher, index_reader=index_reader, doc_id=document_id, tokenized_texts=tokenized_texts)
    # query = doc.get_query(query_size=100, algorithm="kT")
    query = doc.get_mega_query(mega_query_size=10, init_query_size=10, n_docs=2)

    # searching for relevant documents using BM25
    hits = searcher.search(query, k=101)

    # removing the initial document from the retrieved documents and adding the rest to the output list with their score
    return_docs = []
    for i in range(len(hits)):
        if hits[i].docid != document_id:
            return_docs.append([hits[i].docid, hits[i].score])

    # creating the output 2d array:
    output = []
    for i in range(len(return_docs)):
        output.append([topic_number, "Q0", return_docs[i][0], i, return_docs[i][1], "TEAM2"])

    # writing to a file
    for item in output:
        out_file.write(' '.join(map(str, item)) + "\n")

# closing the output file
out_file.close()
