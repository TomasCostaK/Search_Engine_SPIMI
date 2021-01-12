import re
import pandas as pd
import os
import psutil
import sys
import math
import ast

class Indexer:
    def __init__(self,initial_structure={},positional_flag=False):
        self.indexed_words = initial_structure
        self.positional_flag = positional_flag
        self.block_directory = './blocks/'
        self.index_directory = './index/'
        self.collection_size = 0
    
    def getIndexed(self):
        return self.indexed_words

        # we call this extra step, so every term has an idf
    def updateColSize(self, collection_size):
        self.collection_size = collection_size

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
                    string = term + ':' + str(value) + '\n'
                f.write(string)

    def reset_dirs(self):
        try:
            for file in os.listdir(self.block_directory):
                os.unlink(os.path.join(self.block_directory, file))
            for file in os.listdir(self.index_directory):
                os.unlink(os.path.join(self.index_directory, file))
            for file in os.listdir('./tmp/'):
                os.unlink(os.path.join('./tmp/', file))
        except Exception:
            print("Problem resetting directories.")
            sys.exit()

    ## Set of functions for spimi approach
    def create_dirs(self):
        reindex_flag = False
        # blocks
        try:
            os.mkdir(self.block_directory)
        except FileExistsError:
            """
            for file in os.listdir(self.block_directory):
                os.unlink(os.path.join(self.block_directory, file))
            """
            if os.listdir(self.block_directory) != []:
                reindex_flag = True

        # index
        try:
            os.mkdir(self.index_directory)
        except FileExistsError:
            """
            for file in os.listdir(self.index_directory):
                os.unlink(os.path.join(self.index_directory, file))
            """
            if os.listdir(self.index_directory) != []:
                reindex_flag = True
        
        return reindex_flag
    
    def write_info(self, col_size):
        tmp_dir = "./tmp/"
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            for file in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, file))

        with open(tmp_dir + 'info.txt', 'w+') as f:
            f.write(str(col_size))
        f.close()

    def create_block(self, block_nr):
        self.write_index_file(file_output=self.block_directory + '/block' + str(block_nr) + '.txt', idf_flag=False)
        # Clear out index for next block, SPIMI approach
        self.indexed_words = {}

    def merge_blocks(self):
        self.temp_index = {}
        block_files = os.listdir(self.block_directory)
        block_files = [open(self.block_directory+block_file) for block_file in block_files]
        lines = [block_file.readline()[:-1] for block_file in block_files]
        last_term = ""
        index = 0

        # deleting empty files
        for block_file in block_files:
            if lines[index] == "":
                block_files.pop(index)
                lines.pop(index)
            else:
                index += 1

        mem_initial = psutil.virtual_memory().available
        # start looking through words
        while len(block_files) > 0:

            min_index = lines.index(min(lines))
            line = re.split(":",lines[min_index].rstrip('\n'),maxsplit=1)
            current_term = line[0]
            current_postings = line[1]
            
            # we check initially, so we dont put the same term in two diff files
            mem_used = mem_initial - psutil.virtual_memory().available 
            if mem_used > 100000000 and current_term!=last_term: #only for cases bigger than 100Mb 
                self.write_partition_index(mem_used)
                mem_initial = psutil.virtual_memory().available

            if current_term != last_term:
                json_dict = ast.literal_eval(current_postings)
                self.temp_index[current_term] = json_dict
                last_term = current_term
                idf = math.log10(self.collection_size / json_dict['doc_freq'])
                self.temp_index[current_term]['idf'] = idf
            else:
                json_dict = ast.literal_eval(current_postings)
                # update doc_ids
                tmp_dict = self.temp_index[current_term]['doc_ids']
                new_val = {**json_dict['doc_ids'], **tmp_dict} # merging the two dicts
                self.temp_index[current_term]['doc_ids'] = new_val

                # update idf and doc_freq
                self.temp_index[current_term]['doc_freq'] += json_dict['doc_freq']
                self.temp_index[current_term]['col_freq'] += json_dict['col_freq']
                
                # At each step we take the doc_freq so we can calculate updated
                idf = math.log10(self.collection_size / json_dict['doc_freq'])
                self.temp_index[current_term]['idf'] = idf


            lines[min_index] = block_files[min_index].readline()[:-1]

            if lines[min_index] == "":
                    block_files[min_index].close()
                    block_files.pop(min_index)
                    lines.pop(min_index)
        
        # Write the rest of the dict to disk
        self.write_partition_index(mem_used)

    def write_partition_index(self, mem_used):
        ordered_dict = sorted(self.temp_index.items(), key = lambda kv: kv[0])
        smallest_word = ordered_dict[0][0]
        highest_word = ordered_dict[-1][0]
        with open(f"{self.index_directory}{smallest_word}_{highest_word}.txt",'w+') as f:
            for word, value in ordered_dict:
                string = f"{word}:{str(value)}\n"
                f.write(string)
        self.temp_index = {}
        f.close()
