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

#-----------------
# Test zone
#-----------------
# test1 = 'Life is about love'
# result1 = stop_words(test1)
# print(result1)
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

#-----------------
# Test zone
#-----------------
# test1 = 'Life is about love'
# result1 = stemming(test1)
# print(result1)

#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# PREPROCESS - CONTAINS ALL METHODS
#===========================================================================
def preprocess(tree):
    for elem in tree.iter():
        if elem.tag == 'DOC':
            children = list(elem)
            for child in children:
                flag = False
                if child.tag == 'HEADLINE':
                    # TOKENIZATION
                    child.text = tokenisation(child.text)
                    # STOP-WORDS
                    child.text = stop_words(child.text)
                    # STEMMING
                    child.text = stemming(child.text)

                    # this will be added to the body (text) of the document
                    transporter = child.text
                    flag = True
                if child.tag == 'TEXT':
                    # TOKENIZATION
                    child.text = tokenisation(child.text)
                    # STOP-WORDS
                    child.text = stop_words(child.text)
                    # STEMMING
                    child.text = stemming(child.text)

                # adding the headlines to the body of the text
                    if flag == True:
                        child.text = transporter + child.text
                    # print(child.text)
    return tree


#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# INDEXING
#===========================================================================
"""
    Indexing - Basic => each document
"""
def vocab_generator(tree):
    vocabulary = set()

    for elem in tree.iter():
        if elem.tag == 'DOC':
            children = list(elem)
            for child in children:
                if child.tag == 'TEXT':
                    words = child.text.split()
                    for word in words:
                        vocabulary.add(word)
    return vocabulary

# generating vector indicating whether the word occurs
# in the given document or not
def doc_vectors(child_text, vocabulary):
    vector = dict.fromkeys(vocabulary, 0)
    words = child_text.split()
    for word in words:
        vector[word] += 1
    return list(vector.values())

# learning the position of words within the document
def doc_word_positions(child_text, vocabulary):
    word_position = {v: [] for v in vocabulary}
    words = child_text.split()

    for i, word in enumerate(words):
        word_pos = word_position[word]
        word_pos.append(i)
        word_position[word] = word_pos

    return word_position

#===========================================================================
# POSTING
#===========================================================================
# create a dictionary, where every word in a vocab. is linked
# to a list with DocIDs which contain that word!
#
def invert_index_sparse_rep(df):
    invert_indices = dict.fromkeys(list(df.columns),[])
    # print(invert_indices)
    for (columnName, columnData) in df.iteritems():
        # print('Colunm Name : ', columnName)
        # print('Column Contents : ', columnData.values)

        indices = []
        for i in range(0,len(columnData.values)):
            if columnData.values[i] != 0:
                docID_wordFreq = [i,columnData.values[i]]
                indices.append(docID_wordFreq)
        invert_indices[columnName] = indices
        # print(indices)
    return invert_indices

def invert_index_advanced(invert_indices, child_text, vocabulary):

    # this is a dictionary for easier search
    doc_word_pos = [doc_word_positions(text, vocabulary) for text in child_text]
    new_invert_indices = {v: [] for v in vocabulary}

    for index in invert_indices:
        for tuple in invert_indices[index]:
            docID = tuple[0]
            formation2 = [[docID, x] for x in doc_word_pos[docID].get(index)]
            new_invert_indices[index].append(formation2)

    return new_invert_indices

def dictionary_generator(new_invert_indices, number_of_docs):
    new_system = {word: [] for word in new_invert_indices.keys()}
    system = new_invert_indices
    for word in system:
        document_ID = {'docID-' + str(i): [] for i in range(0,number_of_docs)}
        for doc in system.get(word):
            docID = doc[0][0]
            positions = [position[1] for position in doc]
            key1 = 'docID-'+str(docID)
            document_ID[key1] = (positions)
        new_system[word] = document_ID
    return new_system

def master_indexing(tree):
    #vocabulary generator
    vocabulary = vocab_generator(tree)
    # print(vocabulary)

    #get texts of all children in a tree
    child_text = []
    for elem in tree.iter():
        if elem.tag == 'DOC':
            children = list(elem)
            for child in children:
                if child.tag == 'TEXT':
                    child_text.append(child.text)

    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    # Putting every Document into Vector Representation
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    child_text_index = [doc_vectors(text, vocabulary) for text in child_text]

    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    # Formatting the Pandas DataFrame
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    rows = []
    for i in range(1, len(child_text_index) + 1):
        doc = 'D' + str(i)
        rows.append(doc)
    df = pd.DataFrame.from_records(child_text_index)
    df.columns = list(vocabulary)
    df.index = rows
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    # Generating Inverted Indices - with Doc Frequency
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    invert_indices = invert_index_sparse_rep(df)

    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    # Generating Inverted Indices - with Doc Frequency
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    word_location_document = invert_index_advanced(invert_indices, child_text, vocabulary)

    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    # Generating Inverted Indices - with Doc Positions
    # --^--^--^--^--^--^--^--^--^--^--^--^--^--^--^--^
    system = dictionary_generator(word_location_document, len(child_text_index))


    return system

#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# OUTPUT GENERATOR
#===========================================================================
def document_frequency(word_values):
    docs = word_values.keys()
    df = [1 for doc in docs if len(word_values.get(doc))>0]
    return sum(df)

def parse_doc_name(docName: str):
    id = docName[6:]
    return id

def docID_docPosition(word_values):
    docIDs = word_values.keys()
    format = []
    for id in docIDs:
        id_num = int(parse_doc_name(id))
        pos_lst = word_values.get(id)
        format_lst = [[id_num, pos] for pos in pos_lst]
        format.append(format_lst)
    return format

def sort_alphabetically(system):
    ordered = collections.OrderedDict(sorted(system.items()))
    return ordered

def generate_output(system):
    system = sort_alphabetically(system)
    output = open("index.txt", "w+")
    head = list(system.keys())[0]
    for word in system:
        if word != head:
            output.write('\n')
        df = document_frequency(system.get(word))
        format_lst = docID_docPosition(system.get(word))
        output.write(word)
        output.write(':')
        output.write(str(df))

        # print('Word: {}'.format(word))
        # print(format_lst)
        for j in range(0,len(format_lst)):
            if len(format_lst[j]) != 0:
                output.write('\n')
                output.write('\t')
                output.write(str(format_lst[j][0][0]))
                output.write(': ')
                for i in range(0,len(format_lst[j])):
                    output.write(str(format_lst[j][i][1]))
                    if i != len(format_lst[j])-1:
                        output.write(',')

#===========================================================================
# LOAD DATA INTO MEMORY
#===========================================================================
def read_data_to_memory(file_name):
    pass
#===========================================================================
# QUERY PROCESSING
#===========================================================================
# --------------------------------------------------------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# *******************************************
# Boolean Search
# *******************************************
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --------------------------------------------------------------------------
# parsing the boolean queries
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
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --------------------------------------------------------------------------






# Filter Stage 1
def is_one_word(query):
    query = query.lower()
    lst_words = query.split(' ')
    for word in lst_words:
        if word in ['and','or']:
            return False
    return True

# Filter Stage 2
def is_phrase(query):
    if query[0] == '"':
        return True

def prepare_phrase(phrase):
    phrase = re.split('[^a-zA-Z0-9]+', phrase)
    phrase = [elem for elem in phrase if elem != '']
    return phrase

def is_proximity(query):
    if query[0] == '#':
        return True

def prepare_proximity(proximity):
    triple = re.split('[^a-zA-Z0-9]+', proximity)
    triple = [elem for elem in triple if elem != '']
    print(triple)
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

def compound_queries(compound_query):
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

def is_word(elem):
    for char in elem:
        if char.isalpha():
            return True

def get_files(dict_docs):
    files = flatten1(docID_docPosition(dict_docs))
    return files

def flatten1(t):
    return [item for sublist in t for item in sublist]

def flatten2(t):
    return t[0]

def search_files_word(word,system):
    # search for word in the system
    files = []
    if word in list(system.keys()):
        files = get_files(system.get(word))
    files = [file[0] for file in files]
    print('~~~~~~~~~')
    print(files)
    return files

def search_files_phrase(phrase,system):
    word1 = phrase[0]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
    print('Files for word1: {}'.format(files_word1))
    word2 = phrase[1]
    files_word2 = []
    if word2 in list(system.keys()):
        files_word2 = get_files(system.get(word2))
    print('Files for word2: {}'.format(files_word2))
    # compare the lists
    results = []
    if len(files_word1)!=0 and len(files_word2)!=0:
        for doc1 in files_word1:
            for doc2 in files_word2:
                if doc1[0] == doc2[0] and doc1[1]+1 == doc2[1]:
                    results.append(doc1[0])
    return results

def search_files_proximity(proximity,system):
    proximity_indicator = int(proximity[0])
    word1 = proximity[1]
    files_word1 = []
    if word1 in list(system.keys()):
        files_word1 = get_files(system.get(word1))
    print('Files for word1: {}'.format(files_word1))
    word2 = proximity[2]
    files_word2 = []
    if word2 in list(system.keys()):
        files_word2 = get_files(system.get(word2))
    print('Files for word2: {}'.format(files_word2))
    # compare the lists
    results = []
    if len(files_word1) != 0 and len(files_word2) != 0:
        for doc1 in files_word1:
            for doc2 in files_word2:
                if doc1[0] == doc2[0] and abs(doc1[1] - doc2[1]) <= proximity_indicator:
                    results.append(doc1[0])
    return results


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


def filter_queries(queries):
    one_word = [query for query in queries if is_one_word(query)]
    # print('One word only:')
    # print(one_word)

    # this order must stay like this, else proximities are not generated
    words = [query for query in one_word if not is_phrase(query)]
    phrases = [prepare_phrase(query) for query in one_word if is_phrase(query)]
    proximities = [prepare_proximity(query) for query in words if is_proximity(query)]
    words = [query for query in words if not is_proximity(query)]

#===========================================================================================
    comp_queries = [query for query in queries if not is_one_word(query)]
    comp_queries = [compound_queries(query) for query in comp_queries]

    return words, phrases, proximities, comp_queries

def preprocess_querries(file_name):
    queries = read_bool_queries(file_name)
    print('-------------------------------')
    print('Printing queries: ')
    print(queries)
    print('-------------------------------')
    print('Printing list of queries')
    queries = lst_queries(queries)
    print(queries)
    print('-------------------------------')



    words, phrases, proximities, comp_queries = filter_queries(queries)
    # print(queries)
    return words, phrases, proximities, comp_queries


#===========================================================================
#< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >--< | >
#===========================================================================
# MAIN
#==========================================================================
def main(name_of_file):

    print('Parsing the XML tree file...')
    tree = ET.parse(name_of_file)
    # ------------------
    # Preprocess
    # ------------------
    print('Preprocessing the data...')
    tree = preprocess(tree)

    print('Calculating the Indices...')
    system = master_indexing(tree)

    print('Generating output:')
    generate_output(system)

    print('Output successfully generated!')
    print('The indexed documentation of the files can be found in index.txt')

    #=========================================================================
    # SEARCHING
    #=========================================================================
    words, phrases, proximities, comp_queries = preprocess_querries('queries.txt')
    print('Processing the queries: ')
    print('Compounded Queries: ')
    print(comp_queries)
    print('Words: ')
    print(words)
    print('Phrases: ')
    print(phrases)
    print('Proximities: ')
    print(proximities)

    results_words = flatten1([get_result(word,system) for word in words])
    print('\n')
    print('These are the results for singular queries - words: ')
    print(results_words)
    print('\n')

    results_phrases = [get_result(phrase, system) for phrase in phrases]
    print('\n')
    print('These are the results for singular queries - phrases: ')
    print(results_phrases)
    print('\n')

    results_proximities = [get_result(proxy, system) for proxy in proximities]
    print('\n')
    print('These are the results for singular queries - proximities: ')
    print(results_proximities)
    print('\n')

    print('Compound queries: ')
    print(comp_queries)
    reuslts_compound_queries = [compound_query_results(comp_query, system) for comp_query in comp_queries]
    print('\n')
    print('These are the results for comp queries: ')
    print(reuslts_compound_queries)
    print('\n')

    print(system)



main('sample.xml')






