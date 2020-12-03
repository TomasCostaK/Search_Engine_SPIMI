import re
import pandas as pd

class Indexer:
    def __init__(self,initial_structure={}):
        self.indexed_words = initial_structure
    
    def getIndexed(self):
        return self.indexed_words

    def index(self,tokens, idx):

        for token in tokens:
            # Desagragate tuple
            term = token[0]
            idx = token[1]

            if term not in self.indexed_words.keys():
                self.indexed_words[term] = { 'doc_ids': { idx : 1 }, 'idf': None, 'doc_freq': 1, 'col_freq': 1}
            else:
                # get the dictionary that is a value of term
                value_dict = self.indexed_words[term]['doc_ids']
                if idx not in value_dict.keys():
                    value_dict[idx] = 1
                    self.indexed_words[term]['doc_freq'] += 1
                    self.indexed_words[term]['col_freq'] += 1
                else:
                    value_dict[idx] += 1
                    self.indexed_words[term]['col_freq'] += 1
                self.indexed_words[term]['doc_ids'] = value_dict

    def index_query(self,tokens):
        indexed_query = {}
        for token in tokens:
            # Desagragate tuple
            term = token[0]

            if term not in indexed_query.keys():
                indexed_query[term] = 1
            else:
                indexed_query[term] += 1
        return indexed_query