# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:59:55 2021

@author: Lucas Puddifoot, Jort Gutter, Damy Hillen
"""

from pyserini.search.querybuilder import JTermQuery, JTerm
from pyserini.search import SimpleSearcher, querybuilder
from pyserini.index import IndexReader
from dataclasses import dataclass
from document import Document
from tqdm import tqdm
import itertools
import logging
import json
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('bm25_test.log')
file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(file_handler)

INPUT_FILE: str = "./background_linking.txt"
INDEX_LOC: str = "./indexes/wapo"


@dataclass
class Params:
    use_tf_idf: bool = False  # Use tf-idf for the initial query

    rm: str = "bm25"        # "bm25" or "rm3"
    algorithm: str = "kT"   # "kC" or "kT"
    query_size: int = 100
    window_size: int = 4

    use_mega_query: bool = False
    init_query_size: int = 100
    mega_query_size: int = 100
    n_docs: int = 10

    use_relative_query_size = True
    rel_q_size = 0.2    # between 0 and 1
    min_q_size = 70

    query_boosting: bool = False

    output_dir: str = "./results"

    def set_output(self):
        self.output_file: str = f"{self.rm}_{self.algorithm}"\
            f"_qsize={'relative-'+str(self.rel_q_size*100)+'%' if self.use_relative_query_size else self.query_size}"\
            f"{f'_winsize={self.window_size}' if not self.use_tf_idf else ''}"\
            f"{'_mega' if self.use_mega_query else ''}{'_tf-idf' if self.use_tf_idf else ''}"\
            f"{'_boosted' if self.query_boosting else ''}.results"


# make the topic dictionary with keys being topic number and value being document id
def make_topic_dict():
    with open(INPUT_FILE) as f:
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


def filter_hits(hits, document_id, searcher):
    # removing initial document from the retrieved documents and adding the rest to the output list with their score
    return_docs = []
    #for i in range(len(hits)):
    #    if hits[i].docid != document_id:
    #        return_docs.append([hits[i].docid, hits[i].score])
    seen_docs = set()
    for hit in hits:
        doc = json.loads(searcher.doc(hit.docid).raw())
        if hit.docid != document_id and (doc['author'], doc['title']) not in seen_docs:
            return_docs.append([hit.docid, hit.score])
            seen_docs.add((doc['author'], doc['title']))
    return return_docs


def run(params: Params, topics: dict, pbar: tqdm):
    # =========================== Pipeline ===========================
    logger.info(f"Running algorithm with parameters: {params}")

    # create the searcher for the index
    searcher = SimpleSearcher(INDEX_LOC)
    {
        "bm25": searcher.set_bm25,
        "rm3": searcher.set_rm3
    }[params.rm]()

    # create index reader
    index_reader = IndexReader(INDEX_LOC)

    tokenized_texts = {}

    # open the output file
    if not os.path.isdir(params.output_dir):
        logger.debug(f"Creating output directory \'{params.output_dir}\'")
        os.mkdir(params.output_dir)
    out_file = open(os.path.join(params.output_dir, params.output_file), "w")

    # loop through the topic numbers
    for topic_number in topics:
        # getting the document ID for the topic number
        document_id = topics[topic_number]

        logger.info(f"Topic number: {topic_number}, Document: {document_id}")

        # making a query for this document:
        doc = Document(simple_searcher=searcher, index_reader=index_reader, doc_id=document_id,
                       tokenized_texts=tokenized_texts, window_size=params.window_size)

        query_terms, weights = doc.get_mega_query(params) if params.use_mega_query else doc.get_query(params)
        logger.debug(f"Query terms and weights: {weights}")

        if params.query_boosting:
            if params.rm == 'bm25':
                # optional weighing of the terms (does not work with rm3)
                should = querybuilder.JBooleanClauseOccur['should'].value
                boolean_query_builder = querybuilder.get_boolean_query_builder()

                terms = [JTermQuery(JTerm("contents", weighted_term[0])) for weighted_term in weights]
                boosts = [querybuilder.get_boost_query(terms[i], weight[1]) for i, weight in enumerate(weights)]

                for boost in boosts:
                    boolean_query_builder.add(boost, should)
                query = boolean_query_builder.build()
            else:
                logger.warning(f"Retrieval model {params.rm} cannot be used with query boosting!")
                query = query_terms
        else:
            query = query_terms

        logger.debug(f"Generated query: \'{query}\'")

        # searching for relevant documents using the SimpleSearcher
        hits = searcher.search(query, k=150)
        logger.info(f"{len(hits)} hits!")

        # clean results (filter out duplicates and original document)
        return_docs = filter_hits(hits, document_id, searcher)[:100]
        logger.info(f"{len(return_docs)} documents after filtering!")

        # creating the output 2d array:
        output = []
        for i in range(len(return_docs)):
            output.append([topic_number, "Q0", return_docs[i][0], i, return_docs[i][1], "TEAM2"])

        # writing to a file
        for item in output:
            out_file.write(' '.join(map(str, item)) + "\n")

        logger.info(f"{len(output)} line(s) written to {out_file.name}")

        pbar.update(1)

    # closing the output file
    out_file.close()

    # Analyze results and write to result file:
    out_file_name = os.path.join(params.output_dir, params.output_file)
    os.system(f"echo {out_file_name} >> {params.output_dir}/all_results.txt")
    os.system(f"python3 -m pyserini.eval.trec_eval -m map -m P.30 -m ndcg_cut.5 -m recall.30 ./results/qrels.txt "
              f"{out_file_name} | tail -5 >> {params.output_dir}/all_results.txt")
    logger.info(f"Analyzed result written to {params.output_dir}/all_results.txt")


def generate_params() -> list:
    params = [
        ["bm25", "rm3"],
        ["kC", "kT"],
        [50, 100],
        [2, 4, 8],
        [False]
    ]

    params_list = []
    for p in itertools.product(*params):
        rm, algorithm, query_size, window_size, use_mega_query = p
        params_list.append(Params(
            rm=rm,
            algorithm=algorithm,
            query_size=query_size,
            window_size=window_size,
            use_mega_query=use_mega_query
        ))
        params_list[-1].set_output()

    return params_list


def main():
    topics = make_topic_dict()
    params = generate_params()

    logger.info(f"{len(topics)} topic(s) retrieved!")
    logger.info(f"{len(params)} parameter set(s) generated!")

    for i, p in enumerate(params):
        pbar = tqdm(total=len(topics))
        pbar.set_description(f"Parameters {i+1}/{len(params)}")
        run(p, topics, pbar)
        pbar.close()


if __name__ == '__main__':
    main()
