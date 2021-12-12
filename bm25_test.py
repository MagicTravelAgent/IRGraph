# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:59:55 2021

@author: Lucas Puddifoot, Jort Gutter, Damy Hillen
"""
import os.path

from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from dataclasses import dataclass
from document import Document
from tqdm import tqdm
import re


@dataclass
class Params:
    input_file: str = "./background_linking.txt"
    index_loc: str = "./indexes/wapo"

    use_tf_idf: bool = False  # Use tf-idf for the initial query

    rm: str = "bm25"        # "bm25" or "rm3"
    algorithm: str = "kT"   # "kC" or "kT"
    query_size: int = 100
    window_size: int = 3

    use_mega_query: bool = False
    init_query_size: int = 100
    mega_query_size: int = 100
    n_docs: int = 10

    output_dir: str = "./results"
    output_file: str = f"{rm}_{algorithm}_qsize={query_size}{f'_winsize={window_size}' if not use_tf_idf else ''}{'_mega' if use_mega_query else ''}{'_tf-idf' if use_tf_idf else ''}.results"


params = Params()


def make_topic_dict():
    with open(params.input_file) as f:
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
searcher = SimpleSearcher(params.index_loc)
{
    "bm25": searcher.set_bm25,
    "rm3": searcher.set_rm3
}[params.rm]()

# create index reader
index_reader = IndexReader(params.index_loc)

tokenized_texts = {}

# open the output file
if not os.path.isdir(params.output_dir):
    os.mkdir(params.output_dir)
out_file = open(os.path.join(params.output_dir, params.output_file), "w")

# loop through the topic numbers
for topic_number in tqdm(topics):
    # getting the document ID for the topic number
    document_id = topics[topic_number]

    # making a query for this document:
    doc = Document(simple_searcher=searcher, index_reader=index_reader, doc_id=document_id,
                   tokenized_texts=tokenized_texts, window_size=params.window_size)
    query = doc.get_mega_query(params) if params.use_mega_query else doc.get_query(params)

    # searching for relevant documents using the SimpleSearcher
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
