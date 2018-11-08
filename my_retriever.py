import numpy as np

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
    # Method performing retrieval for specified query
        total_doc = []
        for term in index:
            total_doc.extend(index[term])
        self.D = len(list(set(total_doc))) + 1


    def forQuery(self, query):
        # query is the index of query keywords
        # e.g. {a: 1....}
        if self.termWeighting == "binary":
            top_ten = self.binary(query)

        if self.termWeighting == "tf":
            top_ten = self.tf(query)

        if self.termWeighting == "tfidf":
            top_ten = self.tf_idf(query)

        # top_ten = self.all_method(query)

        return top_ten

    def binary(self,query):
        relavent_docid, weight_doc, query_vec, df = self.formate_data(query)
        i_term = 0
        for term in query:
            query_vec[i_term] = 1
            i_doc = 0
            if term in self.index:
                for docid in relavent_docid:
                    weight_doc[i_doc][i_term] = 1 if docid in self.index[term] else 0
                    i_doc += 1
            i_term += 1
        index = self.calculate(weight_doc, query_vec)
        return relavent_docid[index[-10:]]

    def tf(self,query):
        relavent_docid, weight_doc, query_vec,df = self.formate_data(query)
        i_term = 0
        for term in query:
            query_vec[i_term] = query[term]
            i_doc = 0
            if term in self.index:
                for docid in relavent_docid:
                    if docid in self.index[term]:
                        weight_doc[i_doc][i_term] = self.index[term][docid]
                    i_doc+=1
            i_term+=1
        index = self.calculate(weight_doc, query_vec)

        return relavent_docid[index[-10:]]

    def tf_idf(self,query):

        relavent_docid, weight_doc, query_vec,df = self.formate_data(query)
        D = self.D
        i_term = 0
        for term in query:
            idf = np.log10(D/df[i_term])
            query_vec[i_term] = query[term]*idf
            i_doc = 0
            if term in self.index:
                for docid in relavent_docid:
                    if docid in self.index[term]:
                        weight_doc[i_doc][i_term] = self.index[term][docid]*idf
                    i_doc+=1
            i_term+=1

        index = self.calculate(weight_doc, query_vec)
        return relavent_docid[index[-10:]]

    def all_method(self,query):
        #  tf * idf    |D|      df
        relavent_docid, weight_doc, query_vec, df = self.formate_data(query)
        D = self.D
        i_term = 0

        for term in query:
            idf = 0.
            if self.termWeighting == "tfidf":
                idf = np.log10(D / df[i_term])
                query_vec[i_term] = query[term] * idf
            if self.termWeighting == "tf":
                query_vec[i_term] = query[term]
            if self.termWeighting == "binary":
                query_vec[i_term] = 1

            i_doc = 0
            if term in self.index:
                for docid in relavent_docid:
                    if docid in self.index[term]:
                        if self.termWeighting == "tfidf":
                            weight_doc[i_doc][i_term] = self.index[term][docid] * idf
                        if self.termWeighting == "tf":
                            weight_doc[i_doc][i_term] = self.index[term][docid]
                        if self.termWeighting == "binary":
                            weight_doc[i_doc][i_term] = 1 if docid in self.index[term] else 0
                    i_doc += 1
            i_term += 1

        index = self.calculate(weight_doc, query_vec)
        return relavent_docid[index[-10:]]

    def formate_data(self,query):
        relavent_docid = []
        query_vec = []
        df = []

        for term in query:
            query_vec.extend([0])
            if term in self.index:
                keys = self.index[term].keys()
                relavent_docid.extend(keys)
                df.extend([len(keys)+1])
            else:
                df.extend([1])
        query_vec = np.array(query_vec)
        query_vec.dtype = 'float32'
        relavent_docid = np.unique(relavent_docid)
        weight_doc = np.zeros((relavent_docid.shape[0], query_vec.shape[0]))
        return relavent_docid,weight_doc,query_vec,df

    def calculate(self, weight_d, weight_q):
        similarity = np.zeros(weight_d.shape[0])
        for i in range(weight_d.shape[0]):
            b = np.sqrt((weight_d[i]**2).sum()) * np.sqrt((weight_q**2).sum())
            if b == 0 :
                similarity[i] = 0
            else:
                similarity[i] = (weight_d[i] * weight_q).sum() / b
        index = np.argsort(similarity)
        return index



