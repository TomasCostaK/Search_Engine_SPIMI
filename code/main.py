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
    def __init__(self, tokenizer_mode, file='../content/metadata.csv', stopwords_file="../content/snowball_stopwords_EN.txt", chunksize=1000, queries_path='../content/queries.txt' ,rank_mode='bm25', docs_limit=50, positional_flag=False):
        self.tokenizer = Tokenizer(tokenizer_mode, stopwords_file)
        self.indexer = Indexer(positional_flag=positional_flag)
        self.ranker = Ranker(queries_path=queries_path ,mode=rank_mode,docs_limit=docs_limit)
        self.file = file

        # defines the number of lines to be read at once
        self.chunksize = chunksize
        self.block_number = 0

        # tryout for new structure in dict
        self.indexed_map = {}

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
        # here we clean the blocks folder
        output_directory = './blocks/'
        try:
            os.mkdir(output_directory)
        except FileExistsError:
            for file in os.listdir(output_directory):
                os.unlink(os.path.join(output_directory, file))

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
            
                #print("Estimated tokenizing/stemming time: %.4fs" % (toc-tic)) #useful for debugging
                # SPIMI Approach
                block_index = self.indexer.index(tokens, index, positional_flag)
                self.indexer.write_index_file(file_output=output_directory + '/block' + str(self.block_number) + '.txt', idf_flag=False)
                #print("Estimated indexing time: %.4fs" % (toc-tic)) #useful for debugging
                self.block_number += 1

        self.indexed_map = self.indexer.getIndexed()

    def rank(self, analyze_table, tokenizer_mode):
        self.updateIdfs()
        #self.ranker.update(self.docs_length, self.collection_size, self.indexed_map, tokenizer_mode, "../content/snowball_stopwords_EN.txt")
        #self.ranker.process_queries(analyze_table=analyze_table)

    # we call this extra step, so every term has an idf
    def updateIdfs(self):
        for term, value in self.indexed_map.items():
            idf = math.log10(self.collection_size / self.indexed_map[term]['doc_freq'])
            self.indexed_map[term]['idf'] = idf


    def write_index_file(self):
        self.indexer.write_index_file()

    # Questions being asked in work nº1
    def domain_questions(self, time):
        # Question a)
        mem_size = self.calculate_dict_size(self.indexed_map) / 1024 / 1024
        print("A) Estimated process time: %.4fs and spent %.2f Mb of memory" %
            (time, mem_size))

        # Question b)
        vocab_size = len(self.indexed_map.keys())
        print("B) Vocabulary size is: %d" % (vocab_size))

        # Question c)
        # i think we can do this, because these keys only have 1 value, which is the least possible to get inserted into the dict
        ten_least_frequent = [key for (key, value) in sorted(
            self.indexed_map.items(), key=lambda x: x[1]['col_freq'], reverse=False)[:10]]
        # sort alphabetical
        #ten_least_frequent.sort()
        print("\nC) Ten least frequent terms:")
        for term in ten_least_frequent:
            print(term)

        # Question d)
        # i think we can do this, because these keys only have 1 value, which is the least possible to get inserted into the dict
        ten_most_frequent = [key for (key, value) in sorted(
            self.indexed_map.items(), key=lambda x: x[1]['col_freq'], reverse=True)[:10]]
        # sort alphabetical
        #ten_most_frequent.sort()
        print("\nD) Ten most frequent terms:")
        for term in ten_most_frequent:
            print(term)

    # auxiliary function to calculate dict size recursively
    def calculate_dict_size(self, input_dict):
        mem_size = 0
        for key, value in input_dict.items():
            # in python they dont count size, so we have to do it iteratively
            mem_size += sys.getsizeof(value)
            for key2, value2 in value['doc_ids'].items(): 
                mem_size += sys.getsizeof(value2)

        # adding the own dictionary size
        return mem_size + sys.getsizeof(input_dict)

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