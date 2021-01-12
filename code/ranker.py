from numpy import cumsum
from indexer import Indexer
from tokenizer import Tokenizer

import sys
import numpy as np
import os
import time
import math
import collections

class Ranker:
    def __init__(self, queries_path='../content/queries.txt', k1=1.2, b=0.75, mode='tf_idf', docs_limit=50, docs_length={}):
        #values used for bm25, these are the most used defaults
        self.k1 = k1
        self.b = b

        self.indexer = Indexer()
        
        #file location for the queries
        self.queries_path = queries_path

        #type of ranking mode
        self.mode = mode
        
        #limit of docs being analyzed, usually 50 for seeing the table
        self.docs_limit = docs_limit

        # used in ranking to check each documents length, and the average of all docs
        self.docs_length = docs_length

        # used in tf_idf to calculate the square root of all weights
        self.doc_pow = collections.defaultdict(lambda: 0)

        # arrays used to calculate means
        self.mean_precision_array = []
        self.mean_recall_array = []
        self.mean_f_measure_array = []
        self.mean_ap_array = []
        self.mean_ndcg_array = []
        self.mean_latency_array = []

        self.index_directory = './index/'


    def update(self, docs_len, collection_size, tokenizer_mode, stopwords_file):
        self.docs_length = docs_len
        # atributes used in calculus
        if collection_size == 0:
            try:
                file = open('tmp/info.txt',mode='r')
                self.collection_size = int(file.read())
                file.close()
            except Exception:
                print("Error reading previous indexed values\nPlease run: python3 main.py -z")
                sys.exit()
        else:
            self.collection_size = collection_size
        
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
        #update documents length
        self.avdl = sum([ value for key,value in self.docs_length.items()]) / self.collection_size
   
    def process_queries(self, analyze_table=True):
        #Show results for ranking
        with open(self.queries_path,'r') as f:
            query_n = 1
            if analyze_table:
                self.queries_results()
            for query in f.readlines():
                tic = time.time()
                
                if self.mode == 'tf_idf':
                    best_docs = self.rank_tf_idf(query)
                elif self.mode == 'bm25':
                    best_docs = self.rank_bm25(query)
                else:
                    usage()
                    sys.exit(1)

                if not analyze_table:
                    print("Results for query: %s\n" % (query))
                    for doc in best_docs:
                        print("Document: %s \t with score: %.5f" % (doc[0], doc[1]))
                else:
                    docs_ids = [doc_id for doc_id, score in best_docs]
                    # evaluate each query and print a table
                    toc = time.time()
                    self.evaluate_query(query_n, docs_ids, toc-tic)
                
                # update query number
                query_n += 1
        
        if analyze_table:

            # calculate medians, we do it like this, so its easier to read, np was the most efficient way
            mean_precision = np.mean(self.mean_precision_array)
            mean_recall = np.mean(self.mean_recall_array)
            mean_f_measure = np.mean(self.mean_f_measure_array)
            mean_ap = np.mean(self.mean_ap_array)
            mean_ndcg = np.mean(self.mean_ndcg_array)
            mean_latency = np.mean(self.mean_latency_array)
            median_latency = np.median(self.mean_latency_array)

            print("Mean @50: \t %.3f \t\t\t %.3f \t\t\t  %.3f \t\t  %.3f \t\t  %.3f \t  %.0fms " % \
                (mean_precision, mean_recall, mean_f_measure, mean_ap, mean_ndcg, median_latency*1000)
            )
            print("Query throughput: %.3f queries per second" % ( 1 * 1000 / (mean_latency * 1000) ))

    def queries_results(self):
        print("  \t\tPrecision \t\t Recall  	\tF-measure     \tAverage Precision \tNDCG \t\t\t Latency\nQuery #	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50	@10	@20	@50")

    def rank_tf_idf(self, query):
        # declaration of vars to be used in tf.idf
        best_docs = collections.defaultdict(lambda: 0) # default start at 0 so we can do cumulative gains
        N = self.collection_size

        # Special call to indexer, so we can access the term frequency, making use of modularization
        indexed_query = self.indexer.index_query(self.tokenizer.tokenize(query,-1))

        for term,tf_query in indexed_query.items():
            #special treatment, weights at 0

            for file in os.listdir(self.index_directory):
                print("Term: %s, file: %s" % (term, file))
            
            tf_weight = math.log10(tf_query) + 1
            df = self.indexed_map[term]['doc_freq']
            idf = self.indexed_map[term]['idf']

            weight_query_term = tf_weight * idf #this is the weight for the term in the query

            # now we iterate over every term
            for doc_id, doc_id_dict in self.indexed_map[term]['doc_ids'].items():
                tf_doc = doc_id_dict['weight']
                tf_doc_weight = math.log10(tf_doc) + 1
                
                #added step for normalization
                #self.doc_pow[doc_id] += tf_doc_weight ** 2

                score = (weight_query_term * tf_doc_weight)
                best_docs[doc_id] += score

        #normalize after each term
        #for doc_id, score in best_docs.items():
        #    best_docs[doc_id] = score / math.sqrt(self.doc_pow[doc_id])
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:self.docs_limit]


    def rank_bm25(self, query):
        # declaration of vars to be used in tf.idf
        best_docs = collections.defaultdict(lambda: 0) # default start at 0 so we can do cumulative gains
        N = self.collection_size

        # Special call to indexer, so we can access the term frequency, making use of modularization
        indexed_query = self.indexer.index_query(self.tokenizer.tokenize(query,-1))

        for term,tf_query in indexed_query.items():
            #special treatment, weights at 0
            
            for file in os.listdir(self.index_directory):
                smallest_word, highest_word = file.split('.')[0].split('_')
                print(highest_word)
                if term < highest_word and term > smallest_word:
                    print("correct file", file)

            df = self.indexed_map[term]['doc_freq']

            # calculate idf for each term
            idf = self.indexed_map[term]['idf']

            # now we iterate over every term
            for doc_id, doc_id_dict in self.indexed_map[term]['doc_ids'].items():
                tf_doc = doc_id_dict['weight']
                dl = self.docs_length[doc_id]
                score = self.calculate_BM25(df, dl, self.avdl, tf_doc)
                best_docs[doc_id] += idf * score 
        
        most_relevant_docs = sorted(best_docs.items(), key=lambda x: x[1], reverse=True)
        return most_relevant_docs[:self.docs_limit]

    # auxiliary function to calculate bm25 formula
    def calculate_BM25(self, df, dl, avdl, tf_doc):
        N = self.collection_size
        term2 = ((self.k1 + 1) * tf_doc) / ( self.k1 * ((1-self.b) + self.b*dl/avdl) + tf_doc )
        return term2 #since, term1 is idf, and is calculated before

    def evaluate_query(self, query_n, docs_ids, latency):
        #initiate counts at 0
        fp = 0
        tp = 0
        fn = 0

        for i in range(0,3):

            if i==0:
                docs_ids_new = docs_ids[:10]
            elif i==1:
                docs_ids_new = docs_ids[:20]
            elif i==2:
                docs_ids_new = docs_ids[:50]

            #Open queries relevance
            with open('../content/queries.relevance.filtered.txt','r') as q_f:

                # variables for average precision
                doc_counter = 0
                docs_ap = []

                #query_relevance array
                docs_relevance_array = []

                # variables for ndcg
                relevance_ndcg = []

                for q_relevance in q_f.readlines():
                    query_relevance_array = q_relevance.split(" ") # 1st is query number, 2nd is document id, 3rd is relevance
                    
                    if int(query_relevance_array[0]) == query_n:
                        docs_relevance_array.append(query_relevance_array[1]) # append the id, to check for number that dont appear in relevance filter

                        # if relevant and not showing up - FN
                        if int(query_relevance_array[2]) > 0 and query_relevance_array[1] not in docs_ids_new:
                            fn += 1

                        # if showing up but not relevant - FP
                        if int(query_relevance_array[2]) == 0 and query_relevance_array[1] in docs_ids_new:
                            fp += 1
                            # treatment for ndcg
                            relevance_ndcg.append(float(query_relevance_array[2])) 

                        # if showing up and relevant - TP
                        if int(query_relevance_array[2]) > 0 and query_relevance_array[1] in docs_ids_new:
                            tp += 1   
                            try:
                                temp_ap = tp / (fp + tp)
                            except ZeroDivisionError:
                                temp_ap = 0
                            docs_ap.append(temp_ap)      

                            # treatment for ndcg
                            relevance_ndcg.append(float(query_relevance_array[2]))           
                    
                    elif int(query_relevance_array[0]) > query_n:
                        break

                n_outcasts = len(np.setdiff1d(docs_ids_new,docs_relevance_array)) # add the ones that are returned but not on the relevance file
                fp += n_outcasts
                
                # we dont need to update the ndcg, since its 0s to the end, and wouldnt change its values

                # returned values
                # are the special cases necessary?
                try:
                    precision = tp / (fp + tp)
                except ZeroDivisionError:
                    precision = 0
                
                try:
                    recall = tp / ( tp + fn)
                except ZeroDivisionError:
                    recall = 0
                    
                if recall + precision == 0:
                    f_score = 0
                else:
                    f_score = (2 * recall * precision) / (recall + precision)

                # average precision
                try:
                    ap = sum(docs_ap)/len(docs_ap)
                except ZeroDivisionError:
                    ap = 0

                # ndcg
                if len(relevance_ndcg) > 0:
                    ndcg_real = [relevance_ndcg[0]] + [relevance_ndcg[i] for i in range(1,len(relevance_ndcg))]
                    ndcg_real = cumsum(ndcg_real)

                    relevance_ndcg = sorted(relevance_ndcg, reverse=True)
                    ndcg_ideal = [relevance_ndcg[0]] + [relevance_ndcg[i]/(math.log2(i+1)) for i in range(1,len(relevance_ndcg))]
                    ndcg_ideal = cumsum(ndcg_ideal)
                    
                    ndcg = [r / i if i!=0 else 0 for r,i in zip(ndcg_real, ndcg_ideal)][-1]
                else:
                    ndcg = 0

                #do the same but for calculating recall
                if i==0:
                    recall_10 = recall
                    precision_10 = precision
                    f_10 = f_score
                    ap_10 = ap
                    ndcg_10 = ndcg
                elif i==1:
                    recall_20 = recall
                    precision_20 = precision
                    f_20 = f_score
                    ap_20 = ap
                    ndcg_20 = ndcg
                elif i==2:
                    recall_50 = recall
                    precision_50 = precision
                    f_50 = f_score
                    ap_50 = ap
                    ndcg_50 = ndcg

                    # we also add the values to the array of 50 docs
                    self.mean_precision_array.append(precision)
                    self.mean_recall_array.append(recall)
                    self.mean_f_measure_array.append(f_score)
                    self.mean_ap_array.append(ap)
                    self.mean_ndcg_array.append(ndcg)
                    self.mean_latency_array.append(latency)
            
        print("Query: %d  %.3f %.3f %.3f \t %.3f %.3f %.3f \t   %.3f %.3f %.3f \t   %.3f %.3f %.3f \t   %.1f %.1f %.1f \t  %.0fms" % \
            (query_n, precision_10,precision_20,precision_50, recall_10, recall_20, recall_50, f_10, f_20, f_50 \
                ,ap_10,ap_20,ap_50, ndcg_10, ndcg_20, ndcg_50, latency*1000))

        return None

def usage():
    print("Usage: python3 main.py <tokenizer_mode: complex/simple> <chunksize:int> <ranking_mode:tf_idf/bm25> <analyze_flag:boolean>")