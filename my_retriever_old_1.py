import numpy as np

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index      # e.g. {'a': {docid: f_w}, ......}
        self.termWeighting = termWeighting

        total_doc = []
        for term in index:
            total_doc.extend(index[term])
        self.D = len(list(set(total_doc))) + 1


    def forQuery(self, query):
        # query is the index of query keywords
        # e.g. {a: 1....}
        #  tf * idf    |D|   df

        relavent_docid = []
        query_vec = {}
        weight_doc = {}
        D = self.D
        above = {}

        for term in query:
            if term in self.index:
                keys = self.index[term].keys()
                relavent_docid.extend(keys)
        relavent_docid = np.unique(relavent_docid)

        for term in self.index:     # loop all index
            idf = np.log10(D / len(self.index[term].keys()))
            if term in query:  # if the query has this term
                if self.termWeighting == "tf":
                    query_vec[term] = query[term]
                if self.termWeighting == "binary":
                    query_vec[term] = 1
                if self.termWeighting == "tfidf":
                    query_vec[term] = query[term] * idf
            #  calculating the weight of term in query above

            for docid in self.index[term]:        # loop all relavent document
                if docid in relavent_docid:       # if this document has the term
                    if docid not in weight_doc:
                        weight_doc[docid] = {}
                    if self.termWeighting == "tfidf":
                        weight_doc[docid][term] = self.index[term][docid] * idf
                    if self.termWeighting == "tf":
                        weight_doc[docid][term] = self.index[term][docid]
                    if self.termWeighting == "binary":
                        weight_doc[docid][term] = 1 if docid in self.index[term] else 0
                    if docid not in above:
                        above[docid] = 0
                    if term in query:
                        above[docid] += weight_doc[docid][term]*query_vec[term]
            #  calculating the weight of term in document above
        index = self.calculate_similarity(weight_doc, query_vec, above)
        return index


    def calculate_similarity(self, weight_d, weight_q, above):
        similarity = {}
        each_q_d = np.asarray(list(weight_q.values()))
        for docid in weight_d:
            each_w_d = np.asarray(list(weight_d[docid].values()))
            b = np.sqrt((each_w_d**2).sum()) * np.sqrt((each_q_d**2).sum())
            if b == 0 :
                similarity[docid] = 0
            else:
                similarity[docid] = above[docid] / b

        index = np.argsort(list(similarity.values()))

        return np.array(list(similarity.keys()))[index[-10:]]
