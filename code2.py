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
    print('*****')
    print(dict_docs)
    print('*****')
    files = flatten1(docID_docPosition(dict_docs))
    print(files)
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
def get_document_ids(index):
    document_ids = set()
    for word in index:
        for doc_id in index[word]:
            # print(doc_id)
            document_ids.add(int(doc_id))
    return document_ids
def remove_not(query):
    query = query.lower().split(' ')
    query.remove('not')
    # print(query)
    return " ".join(query)
def is_word(elem):
    for char in elem:
        if char.isalpha():
            return True

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
    print(queries_lst)
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
    phrase = [stemming(tokenisation(numbers(case_folding(elem)))) for elem in phrase if elem != '']
    print(phrase)
    return phrase

def prepare_proximity(proximity):
    triple = re.split('[^a-zA-Z0-9]+', proximity)
    triple = [elem for elem in triple if elem != '']
    triple[1] = stemming(tokenisation(case_folding(triple[1])))
    triple[2] = stemming(tokenisation(case_folding(triple[2])))
    return triple

def is_AND(compound_query):
    compound_query = compound_query.split(' ')
    for word in compound_query:
        if word == 'AND':
            return True

def is_OR(compound_query):
    compound_query = compound_query.split(' ')
    for word in compound_query:
        if word == 'OR':
            return True

def prepare_compound_queries(compound_query):
    # compound_query = compound_query.lower()
    flag_AND = is_AND(compound_query)
    flag_OR = is_OR(compound_query)
    if flag_AND:
        print('yessssssssssssssss')
        query = re.split(' AND ', compound_query)
        query = [elem for elem in query if is_word(elem)]
        query = [query[0], 'AND', query[1]]
        return query
    elif flag_OR:
        query = re.split(' OR ', compound_query)
        query = [elem for elem in query if is_word(elem)]
        query = [query[0], 'OR', query[1]]
        return query

"""
    Search for Queries
"""
def search_files_word(word, index):
    # search for word in the system
    files = set()
    # safety check that the word is in my vocab
    if word in list(index.keys()):
        rtrn = get_files(index.get(word))
        for elem in rtrn:
            files.add(elem[0])
    return files

def search_files_phrase(phrase, system):
    print('*****************************')
    word1 = phrase[0]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
        print(files_word1)
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
    print('*****************************')
    return results

def search_files_proximity(proximity, system):
    print(proximity)
    proximity_indicator = int(proximity[0])
    word1 = proximity[1]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
        print(files_word1)
    word2 = proximity[2]
    files_word2 = []
    if word2 in list(system.keys()):
        print(files_word1)
        files_word2 = get_files(system.get(word2))
    # compare the lists
    results = []
    if len(files_word1) != 0 and len(files_word2) != 0:
        for doc1 in files_word1:
            for doc2 in files_word2:
                if doc1[0] == doc2[0] and abs(doc1[1] - doc2[1]) <= proximity_indicator:
                    results.append(doc1[0])
    return results

def get_intersection(query1_result,query2_result):
    intersection = set(query1_result).intersection(set(query2_result))
    return intersection

def get_union(query1_result,query2_result):
    union = set(query1_result).union(set(query2_result))
    return union

def compound_query_results(compound_query_prepared,system):
    flag_AND = compound_query_prepared[1] == 'AND'
    flag_OR = compound_query_prepared[1] == 'OR'

    if flag_AND:
        print('got till here')
        query1_result = execute_query(compound_query_prepared[0],system)
        print('querry1 result: ')
        print(query1_result)
        query2_result = execute_query(compound_query_prepared[2],system)
        print('querry2 result: ')
        print(query2_result)
        result = get_intersection(query1_result,query2_result)
    elif flag_OR:
        query1_result = execute_query(compound_query_prepared[0],system)
        query2_result = execute_query(compound_query_prepared[2],system)
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
        # check if query is negated with NOT
        if is_NOT(query):
            print(query + ' ----------------- YES')
            document_ids = get_document_ids(system)
            query = remove_not(query)
            print(query)
            if is_phrase(query):
                phrase_prepared = prepare_phrase(query)
                phrase_result = (document_ids - set(search_files_phrase(phrase_prepared,system)))
                print(search_files_phrase(phrase_prepared,system))
                print('\nResults: ')
                print(list(phrase_result))
                print('\n')
                return list(phrase_result)
            elif is_proximity(query):
                print('%%%%%%%%')
                print(query)
                proximity_prepared = prepare_proximity(query)
                proximity_result = (document_ids - set(search_files_proximity(proximity_prepared, system)))
                print('\nResults: ')
                print(list(proximity_result))
                print('\n')
                return list(proximity_result)
            else:
                word_prepared = stemming(tokenisation(numbers(case_folding(query))))
                word_results = (document_ids - search_files_word(word_prepared, system))
                print('\nResults: ')
                print(list(word_results))
                print('\n')
                return list(word_results)
        else:
            if is_phrase(query):
                print(query)
                phrase_prepared = prepare_phrase(query)
                phrase_result = search_files_phrase(phrase_prepared,system)
                print('\nResults: ')
                print(list(phrase_result))
                print('\n')
                return phrase_result
            elif is_proximity(query):
                proximity_prepared = prepare_proximity(query)
                print(proximity_prepared)
                proximity_result = search_files_proximity(proximity_prepared, system)
                print('\nResults: ')
                print(list(proximity_result))
                print('\n')
                return proximity_result
            else:
                word_prepared = stemming(tokenisation(numbers(case_folding(query))))
                word_results = search_files_word(word_prepared, system)
                print('\nResults: ')
                print(list(word_results))
                print('\n')
                return word_results
    # Compound:
    else:
        print(query)
        prepared_compound_query = prepare_compound_queries(query)
        print('@@@@@@@@@@@@@@')
        print(prepared_compound_query)
        comp_query_result = list(compound_query_results(prepared_compound_query,system))
        return comp_query_result


def process_querries(file_name,system):
    queries = read_bool_queries(file_name)
    print('\nQueries: ')
    print(queries)
    queries = lst_queries(queries)#[stemming(tokenisation(numbers(case_folding(query)))) for query in lst_queries(queries)]
    print('\nStemmed Queries: ')
    print(queries)
    results = [execute_query(query,system) for query in queries]
    return results

def generate_output_queries(queries_results):
    output = open("results.boolean2.txt", "w+")
    for i in range(0,len(queries_results)):
        for sub_result in queries_results[i]:
            output.write(str(i+1) + ',' + str(sub_result) + '\n')


def main(name_of_file):
    print('Parsing the XML tree file...')
    tree = ET.parse(name_of_file)

    print('Preprocessing the data...')
    documents = document_analysis(tree)

    print('Indexing...')
    index = indexing(documents)
    generate_index_file(index)

    document_ids = get_document_ids(index)
    print('\nThese are document IDs: {}\n'.format(document_ids))

    print('Output successfully generated!')
    print('The indexed documentation of the files can be found in index.txt')

    results = process_querries('queries.boolean.txt', index)
    generate_output_queries(results)

main('trec.sample.xml')





















