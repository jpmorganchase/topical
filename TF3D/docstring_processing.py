"""
Python and Javadocs Handling for docstrings Extraction:

- Extract Docstring : Extract the full docstring without pre processing
- Extract Description : From the lines of quotes, extract the description
- Segment / segment_new : segment list of words, either constrain the string to be partitionable in list of words or not (new)
- normalize_str : Perform Normalization of the list of words
"""

from tqdm import tqdm
import os
import numpy as np
import nltk
import re, string, unicodedata
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import enchant
from sklearn.feature_extraction.text import CountVectorizer
eng_dict = enchant.Dict("en_US")
du= enchant.Dict("en_US")
dg = enchant.Dict("en_GB")

def extract_metafile(index, features):
    path = features[index]['path']
    first_line = features[index]['ast'][1]['line']
    end_line = features[index]['endline']
    count = 1
    with open(path,  encoding="utf8") as file:
        for i in range(1,first_line):
            line = file.readline()
            count += 1
        flag = False
        list_docstring = []
        while count < end_line:
            line = file.readline()
            count += 1
            if flag:
                list_docstring.append(line)
            if '"""' in line:
                if line.count('"""') == 2:
                    list_docstring.append(line)
                elif flag:
                    flag = False
                else:
                    list_docstring.append(line)
                    flag = True
        return list_docstring



def extract_javdoc(index, features):
    path = features[index]['path']
    if index>0:
        first_line = features[index-1]['endline']
    else:
        first_line=0
    end_line = features[index]['ast'][1]['line']
    count = 1

    with open(path,  encoding="utf8") as file:

        for i in range(1,first_line):
            file.readline()
            count += 1
        flag = False
        list_docstring = []
        while count < end_line:
            line = file.readline()
            count += 1
            if flag:
                list_docstring.append(line)
            if '/**' in line:
                list_docstring.append(line)
                flag = True
            if '*/' in line:
                list_docstring.append(line)
                flag = False
        return list_docstring

def extract_description(doc):
    # Remove Indentation, arg and
    new_doc = []
    for el in doc:

        if '---' in el:
            return new_doc[:(-1)]
        
        elif any(ext in el for ext in [':return', 'Ordering:', 'Args', ':param','Parameter:','Parameters', 'parameters:', 'Returns:', ':type', 'Parameters:', 'Accepts:', 'Accepts', '::', ':class:', 'given:' ,':return', 'Return:', 'Ordering:', 'Args', ':param','Parameter:','Parameters', 'parameters:', 'Returns:', ':type', 'Parameters:', 'Accepts:', 'Accepts']):
            return new_doc

        elif ':' in el:
            el = el.split(':')[0]
            el = el.replace('"', '')
            el = el.replace('\n', '')
            el = el.replace(' \\', '')
            el = el.replace('\t', '')
            el = el.replace("`", '')

            el = el.replace('/', '')
            el = el.replace('//', '')
            el = el.replace('/**', '')
            el = el.replace("*", '')
            el = el.replace("/*", '')
            el = el.replace("*/", '')
            i = 0
            if len(el) > 0:
                while (i < len(el)) and (el[i] == ' '):
                    i += 1
                el = el[i:]
            if len(el) > 0:
                new_doc.append(el)
            return new_doc

        elif '.' in el:
            el = el.split('.')[0]
            el = el.replace('"', '')
            el = el.replace('\n', '')
            el = el.replace(' \\', '')
            el = el.replace('\t', '')
            el = el.replace("`", '')

            el = el.replace('/', '')
            el = el.replace('//', '')
            el = el.replace('/**', '')
            el = el.replace("*", '')
            el = el.replace("/*", '')
            el = el.replace("*/", '')
            i = 0
            if len(el) > 0:
                while (i < len(el)) and (el[i] == ' '):
                    i += 1
                el = el[i:]
            if len(el) > 0:
                new_doc.append(el)

            return new_doc
        
        else:
            el = el.replace('"', '')
            el = el.replace('\n', '')
            el = el.replace(' \\', '')
            el = el.replace('\t', '')
            el = el.replace("`", '')

            el = el.replace('/', '')
            el = el.replace('//', '')
            el = el.replace('/**', '')
            el = el.replace("*", '')
            el = el.replace("/*", '')
            el = el.replace("*/", '')
            i = 0
            if len(el) > 0:
                while (i < len(el)) and (el[i] == ' '):
                    i += 1
                el = el[i:]
            if len(el) > 0:
                new_doc.append(el)
    return new_doc




def segment_str(chars, exclude=None):
    """
    Segment a string of chars using the pyenchant vocabulary.
    Keeps longest possible words that account for all characters,
    and returns list of segmented words.

    :param chars: (str) The character string to segment.
    :param exclude: (set) A set of string to exclude from consideration.
                    (These have been found previously to lead to dead ends.)
                    If an excluded word occurs later in the string, this
                    function will fail.
    """
    words = []

    if not chars.isalpha():  # don't check punctuation etc.; needs more work
        return [chars]

    if not exclude:
        exclude = set()
    working_chars = chars
    while working_chars:
        # iterate through segments of the chars starting with the longest segment possible
        for i in range(len(working_chars), 1, -1):
            segment = working_chars[:i]
            if eng_dict.check(segment) and segment not in exclude:
                words.append(segment)
                working_chars = working_chars[i:]
                break
        else:  # no matching segments were found
            if words:
                exclude.add(words[-1])
                return segment_str(chars, exclude=exclude)
            # let the user know a word was missing from the dictionary,
            # but keep the word
            return [chars]
    # return a list of words based on the segmentation
    return words


def segment_str_new(chars, exclude=None):
    """
    Segment a string of chars using the pyenchant vocabulary.
    Keeps longest possible words that account for all characters,
    and returns list of segmented words.

    :param chars: (str) The character string to segment.
    :param exclude: (set) A set of string to exclude from consideration.
                    (These have been found previously to lead to dead ends.)
                    If an excluded word occurs later in the string, this
                    function will fail.
    """
    words = []

    if not chars.isalpha():  # don't check punctuation etc.; needs more work
        return [chars]

    if not exclude:
        exclude = set()
    working_chars = chars
    while working_chars:
        # iterate through segments of the chars starting with the longest segment possible
        for i in range(len(working_chars), 1, -1):
            segment = working_chars[:i]
            if eng_dict.check(segment) and segment not in exclude:
                words.append(segment)
                working_chars = working_chars[i:]
                break
        else:  # no matching segments were found
            try:
                return words + segment_str_new(working_chars[(i + 1):], exclude=None)
            except:
                return words
    # return a list of words based on the segmentation
    return words
stop_words = []

with open(os.path.dirname(os.path.realpath(__file__))+"/stopwords.txt", 'r') as f:
    for line in f:
        stop_words.append(line.rstrip('\n'))

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stopwords"""
    return [el for el in words if el not in stop_words]

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        try:
            lemma = lemmatizer.lemmatize(word, pos=wn.synsets(word)[0].pos())
        except:
            lemma = word
        lemmas.append(lemma)
    return lemmas

def segment(words):
    new_words = []
    for word in words:
        new_word = segment_str(word)
        new_words += new_word
    return new_words


def segment_new(words):
    new_words = []
    for word in words:
        new_word = segment_str_new(word)
        new_words += new_word
    return new_words

def normalize_str(words):
    words = remove_non_ascii(words)
    words = segment(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = lemmatize_verbs(words)
    words = remove_stopwords(words)
    return words


def extract_docstring_file(features,lang="python"):
    
    n = len(features)
    
    docs = []
    
    for i in tqdm(range(n)):
        try:
            if lang=="python":
                el = extract_docstring(i, features)
            else:
                el = extract_javdoc(i, features)
            processed_el = extract_description(el)
        except:
            processed_el = []
        docs.append(processed_el)

    list_docstring = []
    
    for el in docs:
        s = ' '.join(el)
        list_docstring.append(s)

    return list_docstring

def doc_normalization(candidates):
    methods = candidates["methods"].values
    methods = [str(el).split('_') for el in methods]
    methods = [segment_new(el) for el in tqdm(methods)]
    docstrings = candidates["docstring"].values
    docstrings = [str(el).split('_') for el in docstrings]
    docstrings = [segment_new(el) for el in tqdm(docstrings)]
    list_docs = [' '.join(el1) + ' ' + ' '.join(el2) for el1, el2 in zip(methods, docstrings)]
    list_docs_norm = [normalize_str(nltk.word_tokenize(el)) for el in tqdm(list_docs)]
    list_docs_norm_big = [[el for el in element if len(el) >= 2] for element in list_docs_norm]
    return [' '.join(el) for el in list_docs_norm_big]


def doc_selection(corpus, candidates):
    vectorizer = CountVectorizer()
    X_doc = vectorizer.fit_transform(corpus)
    distr = np.array(X_doc.sum(axis = 0)).ravel()
    true_words = np.array([du.check(el) or dg.check(el) or el in ["mnpi", "pii", "pi"]for el in vectorizer.get_feature_names()])
    sel_tags = np.logical_and((distr > 5), true_words)
    selected_tags = np.array(vectorizer.get_feature_names())[sel_tags]
    list_docs_norm_big = [el.split(" ") for el in corpus]
    list_docs_norm_red = [[element for element in el if element in selected_tags] for el in tqdm(list_docs_norm_big)]
    list_length = np.array([len(el) for el in list_docs_norm_red])
    list_docs_norm_red = np.array(list_docs_norm_red)[np.logical_and((list_length <= 20), (list_length > 0))]
    candidates_red = candidates.iloc[np.logical_and((list_length <= 20), (list_length > 0))]
    corpus_red = [' '.join(el) for el in list_docs_norm_red]
    return corpus_red, candidates_red

