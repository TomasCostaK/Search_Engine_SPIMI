from tokenizer import Tokenizer
from indexer import Indexer
from ranker import Ranker
import time

from numpy import cumsum
import sys
import getopt
import operator
import os
import csv
import math
import psutil
import collections

"""
Authors:
Tomás Costa - 89016  
"""

class RTLI:  # Reader, tokenizer, linguistic, indexer
    def __init__(self, tokenizer_mode, file='../content/metadata_small.csv', stopwords_file="../content/snowball_stopwords_EN.txt", chunksize=10000, queries_path='../content/queries.txt' ,rank_mode='bm25', docs_limit=50, positional_flag=False):
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
        self.indexer = Indexer(positional_flag=positional_flag)
        self.ranker = Ranker(queries_path=queries_path ,mode=rank_mode,docs_limit=docs_limit)
        self.file = file

        # defines the number of lines to be read at once
        self.chunksize = chunksize
        self.block_number = 0

        # used in bm25 to check each documents length, and the average of all docs
        self.docs_length = {}

        # collection size
        self.collection_size = 0

    # auxiliary function to generate chunks of text to read
    def gen_chunks(self, reader):
        chunk = []
        for i, line in enumerate(reader):
            if (i % self.chunksize == 0 and i > 0):
                yield chunk
                del chunk[:]  # or: chunk = []
            chunk.append(line)
        yield chunk

    # main function of indexing and tokenizing
    def process(self):

        # Clean dirs
        reindex_flag = self.indexer.create_dirs()

        if not reindex_flag:
            # Reading step
            # We passed the reader to here, so we could do reading chunk by chunk
            with open(self.file, newline='', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for chunk in self.gen_chunks(reader):
                    # Check available memory
                    tokens = []
                    mem = psutil.virtual_memory().available
                    for row in chunk:
                        index = row['cord_uid']
                        # Tokenizer step
                        if row['abstract'] != "":
                            appended_string = row['abstract'] + " " + row['title']
                            tokens += self.tokenizer.tokenize(appended_string, index)

                            self.docs_length[index] = len(tokens)
                            self.collection_size += 1 
                
                    # SPIMI Approach
                    block_index = self.indexer.index(tokens, index, positional_flag)
                    self.indexer.create_block(self.block_number)
                    
                    self.block_number += 1
            
            self.indexer.updateColSize(self.collection_size)
            self.indexer.merge_blocks()
            # we shouldnt load the whole array

        # Here we start evaluating by reading the several index in files
        #self.indexed_map = self.indexer.getIndexed()

    def rank(self, analyze_table, tokenizer_mode):
        #self.ranker.update(self.docs_length, self.collection_size,  tokenizer_mode, "../content/snowball_stopwords_EN.txt")
        #self.ranker.process_queries(analyze_table=analyze_table)
        pass

    def write_index_file(self):
        self.indexer.write_index_file()

def usage():
    print("Usage: python3 main.py \n\t-t <tokenizer_mode: complex/simple> \n\t-c <chunksize:int>\n\t-n <limit of docs returned:int> \n\t-r <ranking_mode:tf_idf/bm25> \n\t-a <analyze_table:boolean>\n\t-p <positional_indexing:boolean>")

if __name__ == "__main__":  

    # work nº1 defaults
    chunksize = 50000
    tokenizer_mode = 'complex'

    # work nº2 defaults
    rank_mode = 'bm25'
    analyze_table = False
    docs_limit = 50

    # work nº3 defaults
    positional_flag = False


    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:c:r:an:p", ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    if len(opts) < 1:
        usage()
        sys.exit()

    for o, a in opts:
        
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o == "-a":
            analyze_table = True
        elif o == "-c":
            try:
                chunksize = int(a)
                assert int(a)>1, "chunksize bigger than 1"
            except:
                usage()
                sys.exit()
            
        elif o == "-n":
            try:
                docs_limit = int(a)
                assert int(a)>1, "docs_limit bigger than 1"
            except:
                usage()
                sys.exit()

        elif o in ["-r", "--ranking"]:
            if a in ["bm25","tf_idf"]:
                rank_mode = a
            else:
                print("Unrecognized ranking mode, use <tf_idf/bm25>\n")
                usage()
                sys.exit(1)

        elif o in ["-t", "--tokenizer"]:
            if a in ["simple","complex"]:
                tokenizer_mode = a
            else:
                print("Unrecognized tokenizer mode, use <simple/complex>\n")
                usage()
                sys.exit(1)
        elif o == "-p":
            positional_flag = True
        else:
            assert False, "unhandled option"
    
    rtli = RTLI(tokenizer_mode=tokenizer_mode,chunksize=chunksize, rank_mode=rank_mode, docs_limit=docs_limit, positional_flag=positional_flag)

    # work nº1 calls
    tic = time.time()
    rtli.process()
    #toc = time.time()

    #rtli.domain_questions(toc-tic)

    # work nº2 calls
    rtli.rank(analyze_table, tokenizer_mode)

    #work nº3 calls
    rtli.write_index_file()
    print("Time : ", time.time()-tic)