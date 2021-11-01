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
"""
    Helper Functions
"""
def get_files(dict_docs):
    files = flatten1(docID_docPosition(dict_docs))
    return files
def flatten1(t):
    return [item for sublist in t for item in sublist]
def docID_docPosition(word_values):
    docIDs = word_values.keys()
    format = []
    for id in docIDs:
        id_num = int(id)
        pos_lst = word_values.get(id)
        format_lst = [[id_num, pos] for pos in pos_lst]
        format.append(format_lst)
    return format
"""
    Parsing & Preprocessing Queries
"""
def read_bool_queries(file_name):
    file_queries = open(file_name, 'r')
    bool_queries = file_queries.readlines()
    return bool_queries

def get_rid_of_number(queries):
    ordered_queries_dict = collections.OrderedDict()
    for i in range(0, len(queries)):
        cnt = len(str(i+1))
        ordered_queries_dict[queries[i][:cnt]] = queries[i][cnt+1:]
    return ordered_queries_dict

def lst_queries(queries):
    queries_lst = []
    for i in range(0, len(queries)):
        cnt = len(str(i+1))
        queries_lst.append(queries[i][cnt+1:-1])
    return queries_lst

"""
    Classifying Queries
"""
def is_single_query(query):
    query = query.lower()
    lst_words = query.split(' ')
    for word in lst_words:
        if word in ['and', 'or']:
            return False
    return True

def is_phrase(query):
    if query[0] == '"':
        return True

def is_proximity(query):
    if query[0] == '#':
        return True

def is_NOT(query):
    query = query.lower()
    lst_words = query.split(' ')
    for word in lst_words:
        if word == 'not':
            return True
    return False

"""
    Preparing Queries
"""
def prepare_phrase(phrase):
    phrase = re.split('[^a-zA-Z0-9]+', phrase)
    phrase = [elem for elem in phrase if elem != '']
    return phrase

def prepare_proximity(proximity):
    triple = re.split('[^a-zA-Z0-9]+', proximity)
    triple = [elem for elem in triple if elem != '']
    return triple

def is_AND(compound_query):
    compound_query = compound_query.lower()
    compound_query = compound_query.split(' ')
    for word in compound_query:
        if word == 'and':
            return True

def is_OR(compound_query):
    compound_query = compound_query.lower()
    compound_query = compound_query.split(' ')
    for word in compound_query:
        if word == 'or':
            return True

def prepare_compound_queries(compound_query):
    compound_query = compound_query.lower()
    flag_AND = is_AND(compound_query)
    flag_OR = is_OR(compound_query)
    if flag_AND:
        query = re.split(' and ', compound_query)
        query = [elem for elem in query if is_word(elem)]
        query = [query[0], 'and', query[1]]
        return query
    elif flag_OR:
        query = re.split(' or ', compound_query)
        query = [elem for elem in query if is_word(elem)]
        query = [query[0], 'or', query[1]]
        return query

"""
    Search for Queries
"""
def search_files_word(word, system):
    # search for word in the system
    files = []
    if word in list(system.keys()):
        files = get_files(system.get(word))
    files = [file[0] for file in files]
    return files

def search_files_phrase(phrase, system):
    word1 = phrase[0]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
    word2 = phrase[1]
    files_word2 = []
    if word2 in list(system.keys()):
        files_word2 = get_files(system.get(word2))
    # compare the lists
    results = []
    if len(files_word1) != 0 and len(files_word2) != 0:
        for doc1 in files_word1:
            for doc2 in files_word2:
                if doc1[0] == doc2[0] and doc1[1] + 1 == doc2[1]:
                    results.append(doc1[0])
    return results

def search_files_proximity(proximity, system):
    proximity_indicator = int(proximity[0])
    word1 = proximity[1]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
    word2 = proximity[2]
    files_word2 = []
    if word2 in list(system.keys()):
        files_word2 = get_files(system.get(word2))
    # compare the lists
    results = []
    if len(files_word1) != 0 and len(files_word2) != 0:
        for doc1 in files_word1:
            for doc2 in files_word2:
                if doc1[0] == doc2[0] and abs(doc1[1] - doc2[1]) <= proximity_indicator:
                    results.append(doc1[0])
    return results

"""
    Get Results for Queries
"""
def get_result(query,system):
    flag_phrase = len(query) == 2
    flag_proximity = len(query) == 3
    flag_word = False
    if flag_phrase:
        phrase = query
        result = search_files_phrase(phrase,system)
        return result
    elif flag_proximity:
        proximity = query
        result = search_files_proximity(proximity,system)
        return result
    else:
        flag_word = True
        word = query
        result = search_files_word(word,system)
        return result

def get_intersection(query1_result,query2_result):
    intersection = set(query1_result).intersection(set(query2_result))
    return intersection

def get_union(query1_result,query2_result):
    union = set(query1_result).union(set(query2_result))
    return union

def compound_query_results(compound_query_prepared,system):
    flag_AND = compound_query_prepared[1] == 'and'
    flag_OR = compound_query_prepared[1] == 'or'

    if flag_AND:
        query1_result = get_result(compound_query_prepared[0],system)
        query2_result = get_result(compound_query_prepared[2],system)
        result = get_intersection(query1_result,query2_result)
    elif flag_OR:
        query1_result = get_result(compound_query_prepared[0],system)
        query2_result = get_result(compound_query_prepared[2],system)
        result = get_union(query1_result,query2_result)
    else:
        result = []
    return result

"""
    Execute Query
"""
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


def process_querries(file_name,system):
    queries = read_bool_queries(file_name)
    queries = [preprocess(query) for query in lst_queries(queries)]
    results = [execute_query(query,system) for query in queries]
    return results

def generate_output_queries(queries_results):
    output = open("results.boolean.txt", "w+")
    for i in range(0,len(queries_results)):
        for sub_result in queries_results[i]:
            output.write(str(i) + ',' + str(sub_result) + '\n')


def main(name_of_file):
    print('Parsing the XML tree file...')
    tree = ET.parse(name_of_file)

    print('Preprocessing the data...')
    documents = document_analysis(tree)

    print('Indexing...')
    index = indexing(documents)
    generate_index_file(index)

    print(index)

    print('Output successfully generated!')
    print('The indexed documentation of the files can be found in index.txt')

    results = process_querries('queries.boolean.txt', index)
    generate_output_queries(results)

main('trec.5000.xml')





















