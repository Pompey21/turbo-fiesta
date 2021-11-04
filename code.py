import xml.etree.ElementTree as ET
import collections
from nltk.stem import PorterStemmer
import re
import math


"""
    The following file is my Information Retrieval System :)
    
    For the purpouses of easier understanding of my code, I have divided the document into three
    components, based on the functionalities we needed to implement:
    
        1. Preprocessing - includes the methods responsible for preprocessing the text data
        2. Indexing - includes the methods for generating the index
        3. Boolean Querying - includes the methods responsible for preparation as well as 
           the execution of the boolean search queries.
        4. Ranked Querying - includes the methods responsible for preparation as well as 
           the execution of the ranked search queries.
        5. Brain - contains the Main() method, which controls the operation of the whole 
           IR system by calling all of the sub-components.
"""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                       1. PREPROCESSING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
    This is the main preprocessing method, which calls all other 
    methods in this sub-component.
"""
def preprocess(text):
    text = stemming(stop_words(tokenisation(text)))
    return text
"""
---------------------    
    TOKENISATION
---------------------    
"""
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

"""
--------------------------    
    STOPWORD REMOVAL
--------------------------
"""
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

"""
------------------
    STEMMING
------------------ 
"""
def stemming(sentance):
    ps = PorterStemmer()
    sentance_lst = sentance.split()
    sentance = ' '.join([ps.stem(x) for x in sentance_lst])
    return sentance


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                         2. INDEXING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def document_analysis(tree):
    documents = [(document.find('DOCNO').text, preprocess(document.find('HEADLINE').text + document.find('TEXT').text).split(' '))
                 for document in tree.iter("DOC")]
    return documents

def document_analysis_dict(documents):
    docs_dict = {}
    for doc in documents:
        docs_dict[doc[0]] = doc[1]
    return docs_dict

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                    3. BOOLEAN QUERYING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
def get_document_ids(index):
    document_ids = set()
    for word in index:
        for doc_id in index[word]:
            document_ids.add(int(doc_id))
    return document_ids
def remove_not(query):
    query = query.lower().split(' ')
    query.remove('not')
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
        query1_result = execute_query(compound_query_prepared[0],system)
        query2_result = execute_query(compound_query_prepared[2],system)
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
            document_ids = get_document_ids(system)
            query = remove_not(query)
            print(query)
            if is_phrase(query):
                phrase_prepared = prepare_phrase(query)
                phrase_result = (document_ids - set(search_files_phrase(phrase_prepared,system)))
                print(search_files_phrase(phrase_prepared,system))
                return list(phrase_result)
            elif is_proximity(query):
                proximity_prepared = prepare_proximity(query)
                proximity_result = (document_ids - set(search_files_proximity(proximity_prepared, system)))
                return list(proximity_result)
            else:
                word_prepared = stemming(tokenisation(numbers(case_folding(query))))
                word_results = (document_ids - search_files_word(word_prepared, system))
                return list(word_results)
        else:
            if is_phrase(query):
                phrase_prepared = prepare_phrase(query)
                phrase_result = search_files_phrase(phrase_prepared,system)
                return phrase_result
            elif is_proximity(query):
                proximity_prepared = prepare_proximity(query)
                proximity_result = search_files_proximity(proximity_prepared, system)
                return proximity_result
            else:
                word_prepared = stemming(tokenisation(numbers(case_folding(query))))
                word_results = search_files_word(word_prepared, system)
                return word_results
    # Compound:
    else:
        prepared_compound_query = prepare_compound_queries(query)
        comp_query_result = list(compound_query_results(prepared_compound_query,system))
        return comp_query_result

def process_bool_querries(file_name, system):
    queries = read_bool_queries(file_name)
    queries = lst_queries(queries)#[stemming(tokenisation(numbers(case_folding(query)))) for query in lst_queries(queries)]
    results = [execute_query(query,system) for query in queries]
    return results

def generate_output_queries(queries_results):
    output = open("results.boolean.txt", "w+")
    for i in range(0,len(queries_results)):
        for sub_result in queries_results[i]:
            output.write(str(i+1) + ',' + str(sub_result) + '\n')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                    4. RANKED QUERYING
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_ranked_queries(file_name):
    file_queries = open(file_name, 'r')
    ranked_queries = lst_queries(file_queries.readlines())
    ranked_queries_preprocessed = [stemming(tokenisation(stop_words(numbers(case_folding(query))))) for query in ranked_queries]
    return ranked_queries_preprocessed

def process_ranked_queries(file_name, index, docs_dict):
    queries = read_ranked_queries(file_name)
    number_of_all_docs = len(docs_dict.keys())

    ranked_queries = [rank_query(query, number_of_all_docs, index, docs_dict) for query in queries]
    generate_output_ranked_queries(ranked_queries)


def rank_query(query, number_of_all_docs, index, docs_dict):
    n = number_of_all_docs
    # words in query
    query = query.split(' ')
    # all relevant docs
    lst_docs = [list(search_files_word(word,index)) for word in query]
    lst_docs = [doc for sub in lst_docs for doc in sub]
    # dictionary (word, df)
    dict_inv_df = {}
    for word in query:
        if index.get(word) == None:
            dict_inv_df[word] = 0
        else:
            dict_inv_df[word] = math.log(n / len(index.get(word)),10)

    lst_docs_scores = []
    for doc in lst_docs:
        tuple_doc_score = get_query_score(doc,query,dict_inv_df,docs_dict)
        lst_docs_scores.append(tuple_doc_score)
    lst_docs_scores.sort()
    if len(lst_docs_scores) > 150:
        lst_docs_scores = lst_docs_scores[:150]
    return  lst_docs_scores


def get_query_score(document,query,dict_inv_df,docs_dict):
    score = round(sum([w_term_doc_score(term,document,dict_inv_df,docs_dict) for term in query]),4)
    return (score,document)


def w_term_doc_score(term,document,dict_inv_df,docs_dict):
    inv_df = dict_inv_df[term]
    tf = get_term_frequncy(term,document,docs_dict)
    result = tf*inv_df
    return result


def get_term_frequncy(term,document,docs_dict):
    doc = docs_dict[str(document)]
    tf = sum([1 for word in doc if word == term])
    result_tf = 1
    if tf != 0:
        result_tf = 1 + math.log(tf,10)
    return result_tf

def generate_output_ranked_queries(ranked_queries):
    output = open("results.ranked.txt", "w+")
    for i in range(0,len(ranked_queries)):
        for article in ranked_queries[i]:
            output.write(str(i+1)+','+str(article[1])+','+str(article[0])+'\n')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#          *******************************************
#                          5. BRAIN
#          *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(name_of_file):
    print('Parsing the XML tree file...')
    tree = ET.parse(name_of_file)

    print('Preprocessing the data...')
    documents = document_analysis(tree)
    docs_dict = document_analysis_dict(documents)

    print('Indexing...')
    index = indexing(documents)
    generate_index_file(index)

    document_ids = get_document_ids(index)
    print('\nThese are document IDs: {}\n'.format(document_ids))

    print('Output successfully generated!')
    print('The indexed documentation of the files can be found in index.txt')

    print('\nProcessing Boolean Queries...')
    results = process_bool_querries('queries.boolean.txt', index)
    results = [sorted(query_results) for query_results in results]
    print('**********')
    print(len(results))
    generate_output_queries(results)
    print('Output successfully generated!')
    print('Results for Boolean Quries can be found in results.boolean.txt')

    print('\nProcessing Ranked Queries...')
    process_ranked_queries('queries.ranked.txt',index, docs_dict)

    # print('\n')
    # print(documents)
    # print(docs_dict)

main('trec.5000.xml')





















