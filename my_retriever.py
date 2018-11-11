import numpy as np

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index      # e.g. {'a': {docid: f_w}, ......}
        self.termWeighting = termWeighting
        # init data and build matrix for all documents
        total_doc_id = []
        for term in index:
            total_doc_id.extend(index[term])
        total_doc_id = np.unique(total_doc_id)
        self.D = total_doc_id.shape[0]
        self.all_doc_vec = np.zeros((total_doc_id.shape[0]+1, len(index.keys())))
        i_term = 0
        for term in self.index:
            idf = np.log10(self.D / len(self.index[term].keys()))
            for docid in self.index[term]:
                if self.termWeighting == "tfidf":
                    self.all_doc_vec[docid][i_term] = self.index[term][docid] * idf
                if self.termWeighting == "tf":
                    self.all_doc_vec[docid][i_term] = self.index[term][docid]
                if self.termWeighting == "binary":
                    self.all_doc_vec[docid][i_term] = 1
            i_term += 1

    def forQuery(self, query):
        # query is the index of query keywords
        # e.g. {a: 1....}
        #  tf * idf    |D|   df
        query_vector = np.zeros(len(self.index.keys()))
        relavent_docid = []
        i_term = 0
        for term in self.index:
            if term in query:  # if the query has this term
                keys = self.index[term].keys()
                relavent_docid.extend(keys)
                if self.termWeighting == "tf":
                    query_vector[i_term] = query[term]
                if self.termWeighting == "binary":
                    query_vector[i_term] = 1
                if self.termWeighting == "tfidf":
                    idf = np.log10(self.D / len(self.index[term].keys()))
                    query_vector[i_term] = query[term] * idf
            i_term += 1
        relavent_docid = np.unique(relavent_docid)
        index = self.calculate_similarity(self.all_doc_vec[relavent_docid], query_vector)
        return relavent_docid[index[-10:]]

    def calculate_similarity(self, weight_d, weight_q):
        similarity = np.zeros(weight_d.shape[0])
        for i in range(weight_d.shape[0]):
            top = (weight_d [i]* weight_q).sum()
            bottom = np.sqrt((weight_d[i] ** 2).sum()) * np.sqrt((weight_q ** 2).sum())
            if bottom == 0:
                similarity[i] = 0
            else:
                similarity[i] = top / bottom
        index = np.argsort(similarity)
        return index
