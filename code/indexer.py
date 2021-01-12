import re
import pandas as pd
import os
import psutil
import ast

class Indexer:
    def __init__(self,initial_structure={},positional_flag=False):
        self.indexed_words = initial_structure
        self.positional_flag = positional_flag
        self.block_directory = './blocks/'
        self.index_directory = './index/'
    
    def getIndexed(self):
        return self.indexed_words

    def index(self,tokens, idx, positional_flag):

        for token in tokens:
            # Desagragate tuple
            term = token[0]
            idx = token[1]
            position = token[2]

            if self.positional_flag == True:
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
            else:
                if term not in self.indexed_words.keys():
                    self.indexed_words[term] = { 'doc_ids': { idx : { 'weight' : 1 }}, 'idf': None, 'doc_freq': 1, 'col_freq': 1}
                else:
                    # get the dictionary that is a value of term
                    value_dict = self.indexed_words[term]['doc_ids']
                    if idx not in value_dict.keys():
                        value_dict[idx] = { 'weight' : 1 }
                        self.indexed_words[term]['doc_freq'] += 1
                        self.indexed_words[term]['col_freq'] += 1
                    else:
                        #already shows up this document
                        value_dict[idx]['weight'] += 1
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
    def write_index_file(self, file_output='../output/indexed_map.txt', idf_flag=True):
        ordered_dict = sorted(self.indexed_words.items(), key = lambda kv: kv[0])   
        with open(file_output,'w+') as f:
            for term, value in ordered_dict:
                if idf_flag:
                    string = term + ": " +  str(value['idf']) + '; ' +  str(value['doc_ids']) + '\n'
                else:
                    string = term + ":{'doc_ids':" +  str(value['doc_ids']) + '}\n'
                f.write(string)


    ## Set of functions for spimi approach
    def create_dirs(self):
        # blocks
        try:
            os.mkdir(self.block_directory)
        except FileExistsError:
            for file in os.listdir(self.block_directory):
                os.unlink(os.path.join(self.block_directory, file))

        # index
        try:
            os.mkdir(self.index_directory)
        except FileExistsError:
            for file in os.listdir(self.index_directory):
                os.unlink(os.path.join(self.index_directory, file))
    
    def create_block(self, block_nr):
        self.write_index_file(file_output=self.block_directory + '/block' + str(block_nr) + '.txt', idf_flag=False)
        # Clear out index for next block, SPIMI approach
        self.indexed_words = {}

    def merge_blocks(self):
        self.temp_index = {}
        block_files = os.listdir(self.block_directory)
        files = [open(self.block_directory+block_file) for block_file in block_files]

        mem_initial = psutil.virtual_memory().available
        # start looking through words
        while True:
            # stopping condition
            if len(files) == 0:
                break
            for file in files:
                try:
                    line = re.split(":",file.readline().rstrip('\n'),maxsplit=1)
                except:
                    continue
                
                # invalid lines
                if line[0] == '':
                    files.remove(file)
                    continue

                if line[0] in self.temp_index.keys():
                    tmp_dict = self.temp_index[line[0]]['doc_ids']
                    new_val = {**ast.literal_eval(line[1])['doc_ids'], **tmp_dict} # merging the two dicts
                    self.temp_index[line[0]]['doc_ids'] = new_val
                else: # we add to dict
                    self.temp_index[line[0]] = ast.literal_eval(line[1])

            mem_used = mem_initial - psutil.virtual_memory().available 
            if mem_used > 100000000: #only for cases bigger than 100Mb 
                ordered_dict = sorted(self.temp_index.items(), key = lambda kv: kv[0])
                smallest_word = ordered_dict[0][0]
                highest_word = ordered_dict[-1][0]
                with open(f"{self.index_directory}{smallest_word}_{highest_word}.txt",'w+') as f:
                    for word, value in ordered_dict:
                        string = f"{word}:{str(value)}\n"
                        f.write(string)
                self.temp_index = {}
                mem_initial = psutil.virtual_memory().available
                print("Memory used:", mem_used)

        
        #print(self.temp_index['wild'])
