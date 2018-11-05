import numpy as np
import pandas as pd

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
    # this is the data from the index file containing stop_list or not and stemming or not
    # e.g. {'a': {docid: f_w}, ......}
        self.termWeighting = termWeighting
    # termWeighting:
    # binary (0,1),
    # tf (Frequency of term in document),
    # tfidf (Frequency in document vs in collection)
        self.query_type = {
            'BINARY': 'binary',
            'TF':'tf',
            'TFIDF' : 'tfidf'
        }
    # Method performing retrieval for specified query

    def forQuery(self, query):
        # query is the index of query keywords
        # e.g. {a: tf....}
        relevant_doc, query_data = self.format_data(query)
        if self.termWeighting == self.query_type['BINARY']:
            results = self.binary_query(query, relevant_doc, query_data)
        elif self.termWeighting == self.query_type['TF']:
            results = self.tf_query(query, relevant_doc, query_data)
        elif self.termWeighting == self.query_type['TFIDF']:
            results = self.tfidf_query(query, relevant_doc, query_data)
        return results.iloc[:10].index.tolist()

    def  binary_query(self, query, relevant_doc, query_data):
        query_data.iloc[0] = 1. # set value for query
        for term in query: # set value for doc
            if term not in self.index:
                relevant_doc[term] = 0.
                continue
            relevant_doc[term].loc[self.index[term].keys()] = 1.
        results = self.calcul_similar(query_data, relevant_doc)
        return results

    def tf_query(self,query, relevant_doc, query_data):
        for term in query:
            query_data[term] = query[term]
            if term not in self.index:
                relevant_doc[term] = 0.
                continue
            relevant_doc[term].loc[self.index[term].keys()] = [self.index[term][k] for k in self.index[term]]
        results = self.calcul_similar(query_data, relevant_doc)
        return results

    def tfidf_query(self,query, relevant_doc, query_data):
        for term in query:
            query_data[term] = query[term]
            if term not in self.index:
                relevant_doc[term] = 0.
                continue
            relevant_doc[term].loc[list(self.index[term].keys())] = [self.index[term][k] for k in self.index[term]]
        # in above calculating the times of each query word in document collection and query module
        d = []
        for k in self.index:
            d.extend(self.index[k])
        d = np.unique(d)    # D = d   |D| = d.shape[0]
        for term in query:
            if term not in self.index:
                continue
            df = (relevant_doc[term]>0).sum() + 1 #  the word is also in query module
            idf = np.log10(d.shape[0]/df)
            query_data[term] = query_data[term]*idf
            relevant_doc[term] = relevant_doc[term]*idf
        results = self.calcul_similar(query_data,relevant_doc)
        return results

    def format_data(self,query):
        # format the matrix of relevant document and query using pandas data structure
        relevant_docid = []
        relevant_doc = {}
        for term in query:
            if term in self.index:
                relevant_doc[term] = self.index[term]
                relevant_docid.extend(self.index[term].keys())
        relevant_docid = np.unique(relevant_docid).tolist()
        relevant_doc_vec = pd.DataFrame(data = np.zeros((len(relevant_docid), len(query.keys()))).tolist(), index =relevant_docid, columns = query.keys() )
        query_vec = pd.DataFrame( data = np.zeros((1, len(query.keys()))).tolist(),  columns = query.keys())
        return relevant_doc_vec, query_vec

    def calcul_similar(self, query_data, document_data):
        query = np.asarray(query_data.loc[0].tolist())
        results = pd.DataFrame(data = np.zeros((document_data.shape[0],1)), index = document_data.index)
        for k in document_data.index:
            doc = np.asarray(document_data.loc[k].tolist())
            result_h = (query*doc).sum()
            result_b = np.sqrt((query**2).sum())*np.sqrt((doc**2).sum())
            if result_b == 0:
                result = 0
            else:
                result = result_h/result_b
            results.loc[k] = result
        return results.sort_values(by = [0], ascending=False)
