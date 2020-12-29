# RI_Assignment2

## Authors
Tomás Costa - 89016  

## Requirements
You need to install the requirements with pip:
```
    pip install -r requirements.txt
```

### Dataset
	https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-12-01/metadata.csv

## How to Run
```python
    cd code
    python3 main.py -h
```

### Usage 
```
Usage: python3 main.py 
	-t <tokenizer_mode: complex/simple> 
	-c <chunksize:int>
	-n <limit of docs returned:int> 
	-r <ranking_mode:tf_idf/bm25> 
	-a <analyze_table:boolean>

```
 * The tokenizer mode specifies if the tokenizer is **simple** or **complex**, and we know the complex one is better to analyze the text, since it deletes pronouns and commonly used words that are not related to the theme of the corpus.  

 * The number_lines defines the amount of lines you want to read at once, we recommend 8000-10000 for this document, since it doesnt slow down a lot but loads way less data into memory.

## Details
The code provided is in the **/code** folder, the answers to the questions are printed by the code with the special option -a
**/content** provides the datasets and texts used.  
**/output** provides the indexed map txt.  

