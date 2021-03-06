from pyserini.search import SimpleSearcher
from nltk.tokenize import word_tokenize
from pyserini.index import IndexReader
from nltk.corpus import stopwords
import networkx as nx
import numpy as np
import itertools
import logging
import json
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('document.log')
file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(file_handler)


class Document:
    def __init__(self, index_reader: IndexReader, simple_searcher: SimpleSearcher, doc_id: str, window_size: int, tokenized_texts: dict = None):
        self.index_reader = index_reader
        self.simple_searcher = simple_searcher
        self.doc_id = doc_id
        self.window_size = window_size
        self.tokenized_texts = tokenized_texts

        if self.doc_id not in self.tokenized_texts:
            logger.info(f"Adding document {self.doc_id} to tokenized texts dictionary.")
            self.doc_tokens = self.tokenize_text(self.doc_id)
            self.tokenized_texts[self.doc_id] = self.doc_tokens

        self.ranking = self.kCT_from_tokens(self.doc_tokens, window_size=self.window_size)
        logger.debug(f"Calculated kC: {self.ranking['kC']}")
        logger.debug(f"Calculated kT: {self.ranking['kT']}")

    def get_text(self, doc_id: str) -> str:
        logger.info(f"Retrieving text of document {doc_id}")
        doc = self.simple_searcher.doc(doc_id)
        raw = doc.raw()
        content = json.loads(raw)
        text = "\n".join(
            [line['content'] for line in content['contents'] if line and line['type'] in ['sanitized_html', 'title'] and line['content']])
        text = re.sub("(<a href=\".*?\">|</a>)", "", text)
        return text

    def tokenize_text(self, doc_id: str) -> list:
        if doc_id in self.tokenized_texts:
            logger.info(f"Document {doc_id} already tokenized, returning saved tokens")
            return self.tokenized_texts[doc_id]

        logger.info(f"Tokenizing document {doc_id}")
        text = self.get_text(doc_id)
        tokens = re.sub("[^a-zA-Z0-9\-]", " ", text.lower())
        tokens = [w for w in word_tokenize(tokens) if w not in stop_words]
        self.tokenized_texts[doc_id] = tokens
        logger.debug(f"Calculated tokens for document {doc_id}: {tokens}")
        return tokens

    @staticmethod
    def kCT_from_tokens(tokenized_text: list, window_size: int = 2) -> (list, list):  # TODO: Add logging (using logger object)
        # Create co-occurrence graph:
        G = nx.Graph()

        for i in range(len(tokenized_text) - (window_size - 1)):  # Avoid index out of bounds
            words = tokenized_text[i:i + window_size]  # Words in window

            for w1 in words:
                for w2 in words:
                    if w1 != w2:  # no self-loops
                        if G.has_edge(w1, w2):
                            G[w1][w2]['weight'] += 1  # update weights on existing edge
                        else:
                            G.add_edge(w1, w2, weight=1)  # add new edge with weight=1

        # Decompose graph using all possible values of k:
        # Dictionaries to hold decomposed k-code and k-truss graphs respectively:
        kcores = {}
        ktrusses = {}

        # k-core:
        for i in itertools.count(start=0):
            icore = nx.algorithms.core.k_core(G, k=i)
            if icore.number_of_nodes() == 0:
                break
            kcores[i] = icore
        # k-truss:
        for i in itertools.count(start=0):
            itruss = nx.algorithms.core.k_truss(G, k=i)
            if itruss.number_of_nodes() == 0:
                break
            ktrusses[i] = itruss

        # find maximum k-values per node (token):
        kC_scores = {}
        kT_scores = {}

        for i in kcores:
            for node in kcores[i].nodes:
                kC_scores[node] = i

        for i in ktrusses:
            for node in ktrusses[i].nodes:
                kT_scores[node] = i

        # find sum of k-values of neighbours per node:
        kC_scores_neighbours = {}
        kT_scores_neighbours = {}

        for node in list(G):
            kC_scores_neighbours[node] = sum([kC_scores[neighbour] for neighbour in
                                              nx.all_neighbors(G, node)])  # sum k-core score of all neighbours
            kT_scores_neighbours[node] = sum([kT_scores[neighbour] for neighbour in
                                              nx.all_neighbors(G, node)])  # sum k-truss score of all neighbours

        # Sort nodes (tokens) by score, descending:
        kC_ranked = sorted(kC_scores_neighbours.items(), key=lambda item: item[1], reverse=True)
        kT_ranked = sorted(kT_scores_neighbours.items(), key=lambda item: item[1], reverse=True)
        logger.info(f"Generated ranking")
        return {
            "kC": kC_ranked,
            "kT": kT_ranked
        }

    def get_query(self, params) -> (str, list):
        logger.info(f"Getting query for {self.doc_id}...")
        if params.use_tf_idf:
            logger.info("Using tf-idf for query")
            return self.get_query_tf_idf(params)

        query_size = max(70, int(params.rel_q_size*len(self.ranking[params.algorithm])))\
            if params.use_relative_query_size else params.query_size

        logger.info("Using graph algorithm for query")

        query = " ".join(
            [w[0] for w in self.ranking[params.algorithm][:query_size]])
        return query, self.ranking[params.algorithm][:query_size]

    def get_query_tf_idf(self, params) -> str:
        # all of these can be retrieved using the index reader
        N = self.index_reader.stats()['documents']
        tf = self.index_reader.get_document_vector(self.doc_id)
        df = {term: (self.index_reader.get_term_counts(term, analyzer=None))[0] for term in tf.keys()}

        # building the idf from these components
        # idf = log((1 + N) / number of documents where the term appears)
        idf = {term: np.log((N + 1) / df[term]) for term in tf.keys()}
        tf_idf = {term: tf[term] * idf[term] for term in tf.keys()}

        top = np.array(sorted([[idf, term] for term, idf in tf_idf.items()], reverse=True)[:params.init_query_size])
        query = ' '.join(top[:, 1])

        return query

    def get_mega_query(self, params) -> (str, list):
        logger.info(f"Getting mega query for {self.doc_id}...")
        q, qw = self.get_query(params)
        hits = self.simple_searcher.search(q, k=params.n_docs)
        logger.info(f"{len(hits)} hits!")
        tokens = []
        for hit in hits:
            tokens.extend(self.tokenize_text(hit.docid))
        logger.info(f"{len(tokens)} tokens found!")
        ranking = self.kCT_from_tokens(tokens, self.window_size)

        return " ".join([w for w, c in ranking[params.algorithm][:params.mega_query_size]]), \
               ranking[params.algorithm][:params.mega_query_size]
