import os
import argparse
import numpy as np
import random
import torch
import gc
from util import get_buckets
import operator
import time
import shutil
import ujson as json
import sys
import lucene
from java.io import File
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
import scipy.sparse
from memory_profiler import profile
import h5sparse

def get_question_list():
    train_record_file = 'train_record.pkl'
    train_buckets = get_buckets(train_record_file)  
    
    question_list = []     
    with open('idx2word.json', 'r') as fh:
        idx2word_dict = json.load(fh)
    with open('word2idx.json', 'r') as fh:
        word2idx_dict = json.load(fh)
    
    for i in range(len(train_buckets[0])):
        ques_idxs = train_buckets[0][i]['ques_idxs']
        ques_idxs = ques_idxs[ques_idxs > 1]   # "0":"--NULL--","1":"--OOV--"
        ques_words = ' '.join([idx2word_dict[str(idx)] for idx in ques_idxs if idx > 0])
        question_list.append(ques_words)

    return question_list

def addDoc(indexwriter, question):
    doc = Document()
    doc.add(TextField("question", question, Field.Store.YES))
    indexwriter.addDocument(doc)     
        
def lucene_indexDir():
    index_path_str = os.getcwd() + '/index'
    index_path = File(index_path_str).toPath()
    indexDir = SimpleFSDirectory.open(index_path)
    return indexDir

def index_questions():
    question_list = get_question_list()
    lucene.initVM()
    indexDir = lucene_indexDir()
    analyzer = EnglishAnalyzer()

    writerConfig = IndexWriterConfig(analyzer)
    writer = IndexWriter(indexDir, writerConfig)   

    T_before_index = time.time()
    for question in question_list:
        addDoc(writer, question);
    writer.close();                         #writeindex to file
    T_after_index = time.time()
    T = T_after_index - T_before_index
    print("index takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T)))

@profile      
def query_questions():
    question_list = get_question_list()
    lucene.initVM()
    indexDir = lucene_indexDir()
    analyzer = EnglishAnalyzer()
    
    question_num = len(question_list)
    matrix_row = np.ones(question_num * question_num) * -1
    matrix_col = np.ones(question_num * question_num) * -1
    matrix_data = np.ones(question_num * question_num) * -1
    
    reader = DirectoryReader.open(indexDir) #read index from file
    searcher = IndexSearcher(reader)
    docCount = searcher.collectionStatistics("question").docCount()
    assert docCount == question_num
    
    #threshold = 0.3
    k = 0
    T_pre = time.time()
    for i, question in enumerate(question_list):
        
        query = QueryParser("question", analyzer).parse(QueryParser.escape(question))
        hits = searcher.search(query, question_num)    
        if(len(hits.scoreDocs) == 0):  # no match
            continue
        else:
            for hit in hits.scoreDocs:
                matrix_row[k] = i
                matrix_col[k] = hit.doc
                matrix_data[k] = hit.score  # can considered as similarity/distance from this unlabeled question to labeled questions set, max similarity ≈ min distance 
                k+=1
                # if(hit.score > threshold):    
                    # matrix_row.append(i)
                    # matrix_col.append(hit.doc)
                    # matrix_data.append(hit.score)  
              
        if ((i+1) % 20000 == 0 or i==len(question_list)-1): 
            T_current = time.time()
            T = T_current - T_pre
            if ((i+1) % 20000 == 0 ):
                print("query 200000 questions takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T))) 
                # lucene_similarity_file = 'lucene_similarity_matrix_' + str((i+1) // 20000) +'.json'
            else:
                print("query", len(question_list) % 20000, "questions takes time: ", time.strftime("%Hh %Mm %Ss", time.gmtime(T))) 
                # lucene_similarity_file = 'lucene_similarity_matrix_' + str(i // 20000 + 1) +'.json'
            T_pre = time.time()
    
    matrix_row = matrix_row[:k]
    matrix_col = matrix_col[:k]
    matrix_data = matrix_data[:k]
    
    # sanity check
    print(len(matrix_row))
    print(len(matrix_col))
    print(len(matrix_data))
    print(matrix_row[:10])
    print(matrix_col[:10])
    print(matrix_data[:10])
    
    del question_list
    del query
    del hits
    del searcher
    del reader
    gc.collect()
    
    lucene_similarity_matrix = scipy.sparse.csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(question_num, question_num))
    with h5sparse.File("lucene_sparse.h5") as h5f:
        h5f.create_dataset('similarity_matrix', data=lucene_similarity_matrix)
            
    # return (matrix_row, matrix_col, matrix_data)

    # memory overflow when process after loading, because the loaed matrix takes a lot memory
    # lucene_similarity_file = 'lucene_similarity_matrix.npz'   
    # np.savez_compressed(lucene_similarity_file, row=matrix_row, col=matrix_col, data=matrix_data)
    
    # too slow to save
    # lucene_similarity_file = 'lucene_similarity_sparse_matrix.npz'     
    # lucene_similarity_matrix = scipy.sparse.csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(question_num, question_num))
    # scipy.sparse.save_npz(lucene_similarity_file, lucene_similarity_matrix)
    
    # memory overflow when saving lists
    # lucene_sparse_matrix = {'row':matrix_row, 'col':matrix_col, 'data':matrix_data}
    # with open(lucene_similarity_file, 'w') as f:
    #     json.dump(lucene_sparse_matrix, f) 
    
if __name__ == '__main__':
    #index_questions()
    query_questions()
    

     