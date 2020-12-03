import re
import Stemmer
class Tokenizer:
    def __init__(self,tokenizer_mode,stopwords_file):
        self.stemmer = Stemmer.Stemmer('english')
        self.tokenizer_mode = tokenizer_mode
        if tokenizer_mode == "complex":
            text = open(stopwords_file,'r')
            self.stopwords = [word.strip() for word in text.readlines()]
        else:
            self.stopwords = []

    
    # Function to read any text and add it to the word dictionary of the Tokenizer
    def tokenize(self,input_string,index):
        final_tokens = []

        # we do the simple tokenizer
        if self.tokenizer_mode == "simple":
            tokens = re.sub("[^a-zA-Z]+"," ",input_string).lower().split(" ")
            tokens = [ (token, tokens.index(token)) for token in tokens ]

        # we go into the complex tokenizer
        else:
            tokens = re.sub("[^0-9a-zA-Z]+"," ",input_string).lower().split(" ") # Make some changes here, having into account that this is a biomedical corpus
            # Snowball stemmer - PyStemmer implementation
        
            tokens = self.stemmer.stemWords(tokens)


        # Iterate over each word in line 
        for token in tokens: 
            # Disregard words with less than 3 chars, or if they are a stopword
            if len(token[0])<3 or token[0] in self.stopwords: 
                continue

            # if it passes the condition, we shall add it to the final_tokens
            final_tokens.append((token[0],index, token[1])) #token 1 represents its position

        return final_tokens