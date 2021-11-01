import string
import xml.etree.ElementTree as ET
import collections
from nltk.stem import PorterStemmer
import pandas as pd
import re


# TASK 1: is to preprocess the data as it was done in lab 1
    # Be sure that you have your preprocessing module ready (revise: lab 1), then apply it to the collections.
    # if you didn't have it done, then at least get the tokeniser and casefolding ready for this lab

#========================================================
# TOKENISATION
#========================================================
"""
    Case Folding
"""
def case_folding(sentance):
    sentance = sentance.lower()
    return sentance

"""
    Numbers Handling
"""
def numbers(sentance):
    numbers = list(range(0, 10))
    numbers_strs = [str(x) for x in numbers]

    for number in numbers_strs:
        sentance = sentance.replace(number, '')
    return sentance

"""
    Tokenisation
"""
# splitting at not alphabetic characers
def tokenisation(sentance):
    sentance_list = re.split('\W+', sentance)
    sentance_list_new = []
    for word in sentance_list:
        word_new = case_folding(numbers(word))
        sentance_list_new.append(word_new)
    return ' '.join(sentance_list_new)

#========================================================
# STOPWORDS REMOVAL
#========================================================
def stop_words(sentance):
    stop_words = open("stop-words.txt", "r").read()
    stop_words = set(stop_words.split('\n'))

    sentance_lst = sentance.split()
    clean_sentance_lst = []

    for word in sentance_lst:
        if word not in stop_words:
            clean_sentance_lst.append(word)
    sentance = ' '.join(clean_sentance_lst)
    return sentance

#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# STEMMING
#===========================================================================
def stemming(sentance):
    ps = PorterStemmer()
    sentance_lst = sentance.split()
    sentance = ' '.join([ps.stem(x) for x in sentance_lst])
    return sentance

#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# PREPROCESS - CONTAINS ALL METHODS
#===========================================================================
def preprocess(text):
    text = stemming(stop_words(tokenisation(text)))
    return text

#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# *******************************************
# INDEXING
# *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def document_analysis(tree):
    documents = [(document.find('DOCNO').text, preprocess(document.find('HEADLINE').text + document.find('TEXT').text).split(' '))
                 for document in tree.iter("DOC")]
    return documents

def indexing(documents):
    index = {}
    for document in documents:
        for (ind,word) in enumerate(document[1]):
            if word not in index:
                index.update({word : {document[0] : [ind]}})
            else:
                if document[0] not in index[word]:
                    index[word][document[0]] = [ind]
                else:
                    index[word][document[0]].append(ind)
    index = collections.OrderedDict(sorted(index.items()))
    return index

def generate_index_file(index):
    output = open("index.txt","w+")
    for word in index:
        output.write(word+':'+str(len(index[word]))+'\n')
        for occurance in index[word]:
            output.write('\t'+occurance+':'+','.join([str(elem) for elem in index[word][occurance]])+'\n')
#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
def is_single_query(query):
    query = query.lower()
    lst_words = query.split(' ')
    for word in lst_words:
        if word in ['and', 'or']:
            return False
    return True

def execute_query(query,system):
    # check is it is a singular or a compound query
    single_query = is_single_query(query)
    # Singular:
    if single_query:
        if is_phrase(query):
            phrase_prepared = prepare_phrase(query)
            phrase_result = search_files_phrase(phrase_prepared,system)
            return phrase_result
        elif is_proximity(query):
            proximity_prepared = prepare_proximity(query)
            proximity_result = search_files_proximity(proximity_prepared, system)
            return proximity_result
        else:
            word_prepared = query
            word_results = search_files_word(word_prepared, system)
            return word_results
    # Compound:
    else:
        prepared_compound_query = prepare_compound_queries(query)
        comp_query_result = list(compound_query_results(prepared_compound_query,system))
        return comp_query_result




def main(name_of_file):
    print('Parsing the XML tree file...')
    tree = ET.parse(name_of_file)

    print('Preprocessing the data...')
    documents = document_analysis(tree)

    print('Indexing...')
    index = indexing(documents)
    generate_index_file(index)

main('trec.5000.xml')





















