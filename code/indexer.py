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
            position = token[2]


            if term not in self.indexed_words.keys():
                self.indexed_words[term] = { 'doc_ids': { idx : { 'weight' : 1 , 'positions' : [position] }}, 'idf': None, 'doc_freq': 1, 'col_freq': 1}
            else:
                # get the dictionary that is a value of term
                value_dict = self.indexed_words[term]['doc_ids']
                if idx not in value_dict.keys():
                    value_dict[idx] = { 'weight' : 1 , 'positions' : [position] }
                    self.indexed_words[term]['doc_freq'] += 1
                    self.indexed_words[term]['col_freq'] += 1
                else:
                    #already shows up this document
                    value_dict[idx]['weight'] += 1
                    value_dict[idx]['positions'].append(position)
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

    # function to write indexed terms to file, in a similar output to the one requested
    def write_index_file(self, file_output='../output/indexed_map.txt'):
        with open(file_output,'w+') as f:
            for term, value in self.indexed_words.items():
                string = term + ": " +  str(value['idf']) + '; ' +  str(value['doc_ids']) + '\n'
                f.write(string)