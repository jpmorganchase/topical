
import os
from TF3D.docstring_processing import *
from TF3D.utils import create_docstring_dataframe
from sklearn.feature_extraction.text import CountVectorizer
import enchant
from sklearn.decomposition import TruncatedSVD
du= enchant.Dict("en_US")
dg = enchant.Dict("en_GB")
import numpy as np
DIR = 'features' # Folder where you have your features.msgpack file


def save_msgpack_features(repository_path, features):
    import msgpack
    with open(repository_path + "features.msgpack", "wb") as outfile:
        packed = msgpack.packb(features)
        outfile.write(packed)

def get_vector(features):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorout = CountVectorizer(binary=True)

def prepare_data_new(features, dataset_path, langdoc='python'):
    #os.chdir(dataset_path) # to remove
    docs = extract_docstring_file(features, lang=langdoc)
    #packages = extract_packages(features)
    candidates = create_docstring_dataframe(features, docs)
    list_docs = doc_normalization(candidates) # needs to have punkt from nltk.
    list_docs_norm = [el.split() for el in list_docs]
    list_docs_norm_big = [[el for el in element if len(el) > 2] for element in list_docs_norm] #Select tokens with more than two characters
    corpus = [' '.join(el) for el in list_docs_norm_big]

    vectorizer = CountVectorizer()
    X_doc = vectorizer.fit_transform(corpus)
    distr = np.array(X_doc.sum(axis=0)).ravel()

    #data preprocessing
    (distr <= 5).sum()
    true_words = np.array([du.check(el) or dg.check(el) for el in vectorizer.get_feature_names()]) #Consider only the tokens having english meaning
    sel_tags = np.logical_and((distr > 5), true_words)
    # Select tags present more than a certain amount of time
    selected_tags = np.array(vectorizer.get_feature_names())[sel_tags]
    list_docs_norm_red = [[element for element in el if element in selected_tags] for el in tqdm(list_docs_norm_big)]
    list_length = np.array([len(el) for el in list_docs_norm_red])
    list_docs_norm_red = np.array(list_docs_norm_red)[np.logical_and((list_length <= 20), (list_length > 0))]
    candidates_red = candidates.iloc[np.logical_and((list_length <= 20), (list_length > 0))]
    corpus_red = [' '.join(el) for el in list_docs_norm_red]
    candidates_red['normalized_docs'] = corpus_red
    candidates_red = candidates_red[candidates_red['features'].apply(lambda r: len(r) > 150)] # Consider only big functions
    #candidates_red = candidates_red[candidates_red["docstring"].apply(lambda r: len(r) > 0)] # Consider only functions with docstring
     
    # further removal for possible 3rd party files.
    candidates_red = candidates_red[~candidates_red['path'].str.contains('site-package')]

    return candidates_red


def prepare_data(features, langdoc='python'):
    #os.chdir(r'/Users/V773117/Code_Tagging/') # to remove
    vectorout=CountVectorizer(binary = True)
    featurestring = []
    for i in features:
        featurestring.append(" ".join([str(j) for j in i['features']]))
    X=vectorout.fit_transform(featurestring)
    docs = extract_docstring_file(features, lang=langdoc)
    candidates = create_docstring_dataframe(features, docs)
    #candidates = candidates[~candidates['docstring'].isnull()]
    #candidates =candidates[candidates['docstring'] != '']
    list_docs = doc_normalization(candidates) # needs to have punkt from nltk.
    candidates.head()

    list_docs_norm = [el.split() for el in list_docs]
    list_docs_norm_big = [[el for el in element if len(el) > 2] for element in list_docs_norm] #Select tokens with more than two characters
    corpus = [' '.join(el) for el in list_docs_norm_big]

    vectorizer = CountVectorizer()
    X_doc = vectorizer.fit_transform(corpus)
    distr = np.array(X_doc.sum(axis = 0)).ravel()

    #data preprocessing
    (distr <= 5).sum()
    true_words = np.array([du.check(el) or dg.check(el) for el in vectorizer.get_feature_names()]) #Consider only the tokens having english meaning
    sel_tags = np.logical_and((distr > 5), true_words)
    # Select tags present more than a certain amount of time
    selected_tags = np.array(vectorizer.get_feature_names())[sel_tags]
    list_docs_norm_red = [[element for element in el if element in selected_tags] for el in tqdm(list_docs_norm_big)]
    list_length = np.array([len(el) for el in list_docs_norm_red])
    list_docs_norm_red = np.array(list_docs_norm_red)[np.logical_and((list_length <= 20), (list_length > 0))]
    candidates_red = candidates.iloc[np.logical_and((list_length <= 20), (list_length > 0))]
    corpus_red = [' '.join(el) for el in list_docs_norm_red]
    candidates_red['normalized_docs'] = corpus_red
    candidates_red = candidates_red[candidates_red['features'].apply(lambda r: len(r) > 150)] # Consider only big functions
    #candidates_red = candidates_red[candidates_red["docstring"].apply(lambda r: len(r) > 0)] # Consider only functions with docstring
    candidates_red.head()
   
    return candidates_red, X

def fit_and_svd(candidates_red, X,k=0.8,k2=0.1,svd_comp=50):
    # divide to test by repos:
    from collections import Counter
    repocount = []
    import random
    for i, pa in candidates_red.iterrows():
        p = os.path.normpath(pa['path'])
        repocount.append(p.split(os.sep)[5] + os.sep + p.split(os.sep)[6])
    counter = Counter(repocount)
    items = [i for i in counter.keys()]
    random.shuffle(items)
    l= int(k*len(items))
    l2=int(k2*len(items))

    trainmask = candidates_red['path'].str.contains('|'.join(items[0:l]))
    valmask = candidates_red['path'].str.contains('|'.join(items[l:l+l2]))
    testmask = candidates_red['path'].str.contains('|'.join(items[l+l2:]))

    ca_train = candidates_red[trainmask]
    ca_val = candidates_red[valmask]
    ca_test = candidates_red[testmask]
    # Truncated SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=svd_comp)
    print(X[ca_train.index, :].shape)
    X_svd = svd.fit_transform(X[ca_train.index, :])
    print(svd.explained_variance_ratio_.sum())
    X_svd_test = svd.transform(X[ca_test.index, :])
    X_svd_val = svd.transform(X[ca_val.index, :])

    vectorizer_red = CountVectorizer(binary=True)
    Yvals = ca_train['normalized_docs'].values
    X_doc_red = vectorizer_red.fit_transform(Yvals)

    Yvals_test = ca_test['normalized_docs'].values
    X_doc_red_test = vectorizer_red.transform(Yvals_test)
    Yvals_val = ca_val['normalized_docs'].values
    X_doc_red_val = vectorizer_red.transform(Yvals_val)

    return ca_train, X_doc_red, X_svd, ca_val, X_doc_red_val, X_svd_val, ca_test, X_doc_red_test, X_svd_test, vectorizer_red


def get_training_val_test_items(df, dataset_info, k=0.8,k2=0.1, IdentiferLevel=-1,size=10):
    from collections import Counter
    cand_paths=df.copy()
    cand_paths['path']=df['path'].apply(lambda x: x.split(os.sep)[IdentiferLevel])
    df['reponame']=cand_paths['path']
    cand_paths=cand_paths.groupby('path').count()
    # divide to train and test by repos:
    # from collections import Counter
    repocount = []
    repo_topics = []
    import random
    for repo_name, pa in cand_paths.iterrows():
        if repo_name in list(dataset_info['full_name']):
            repo_topics.append(dataset_info[dataset_info['full_name']==repo_name]['featured_topics'].values[0])
            repocount.append(repo_name)
        else:
            print("doesn't exist repo: ", repo_name)

    # create a dict of each unique entry and the associated indices
    a=np.array(repo_topics)
    idx = {v: np.where(a == v)[0].tolist() for v in np.unique(a)}
    cand_paths['topics'] = repo_topics
    limited_ind=[]
    for id in idx.values():
        limited_ind.extend(id[0:size])
    repocount=[repocount[i] for i in limited_ind]
    repo_topics=[repo_topics[i] for i in limited_ind]

    df=df.assign(topics='topic')
    for i in range(len(repocount)):
        df.loc[df['reponame']==repocount[i], 'featured_topics'] = repo_topics[i]
    

    counter = Counter(repocount)
    items = list(counter.keys())
    random.shuffle(items)

    l = int(k * len(items))
    l2 = int(k2 * len(items))

    trainmask = df['path'].str.contains(r'{}'.format('|'.join(items[0:l])))
    valmask = df['path'].str.contains('|'.join(items[l:l + l2]))
    testmask = df['path'].str.contains('|'.join(items[l + l2:]))

    ca_train = df[trainmask]
    ca_val = df[valmask]
    ca_test = df[testmask]
    # Masks are returned but not used outside of this method
    return ca_train, ca_val, ca_test, items[0:l], items[l:l + l2], items[l + l2:]

def divide_data_according_to_items(df, dataset_info, items_train, items_val, items_test, IdentiferLevel):
    from collections import Counter
    cand_paths=df.copy()
    cand_paths['path']=df['path'].apply(lambda x: x.split(os.sep)[IdentiferLevel])
    df['reponame']=cand_paths['path']
    cand_paths=cand_paths.groupby('path').count()
    # divide to train and test by repos:
    # from collections import Counter
    repocount = []
    repo_topics = []
    import random
    for repo_name, pa in cand_paths.iterrows():
        if repo_name in list(dataset_info['full_name']):
            repo_topics.append(dataset_info[dataset_info['full_name']==repo_name]['featured_topics'].values[0])
            repocount.append(repo_name)
        else:
            print("doesn't exist repo: ", repo_name)
    df=df.assign(topics='topic')
    for i in range(len(repocount)):
        df.loc[df['reponame']==repocount[i], 'featured_topics'] = repo_topics[i]
    

    trainmask = df['path'].str.contains(r'{}'.format('|'.join(items_train)))
    valmask = df['path'].str.contains('|'.join(items_val))
    testmask = df['path'].str.contains('|'.join(items_test))

    ca_train = df[trainmask]
    ca_val = df[valmask]
    ca_test = df[testmask]

    return ca_train, ca_val, ca_test

def fit_and_svd_features(X, ca_train,ca_val, ca_test, svd_comp=50):
    # Truncated SVD
    svd = TruncatedSVD(n_components=svd_comp)

    X_svd = svd.fit_transform(X[ca_train.index, :])
    print(svd.explained_variance_ratio_.sum())

    X_svd_test = svd.transform(X[ca_test.index, :])
    X_svd_val = svd.transform(X[ca_val.index, :])

    return X_svd, X_svd_val, X_svd_test

def fit_docs(ca_train,ca_val, ca_test, svd_comp=None):
    # Truncated SVD
    vectorizer_red = CountVectorizer(binary=True)
    Y = ca_train['normalized_docs'].values
    X_doc_red = vectorizer_red.fit_transform(Y)

    Y_test = ca_test['normalized_docs'].values
    X_doc_red_test = vectorizer_red.transform(Y_test)
    Y_val = ca_val['normalized_docs'].values
    X_doc_red_val = vectorizer_red.transform(Y_val)

    if svd_comp is not None:
        svd = TruncatedSVD(n_components=svd_comp)

        X_doc_red = svd.fit_transform(X_doc_red)
        X_doc_red_val = svd.transform(X_doc_red_val)
        X_doc_red_test = svd.transform(X_doc_red_test)

    return X_doc_red, X_doc_red_val, X_doc_red_test, vectorizer_red

def fit_docs_from_docmatrix(X_doc, ca_train, ca_val=None, ca_test=None, svd_comp=None):
    # Truncated SVD
    from sklearn.decomposition import TruncatedSVD
    X_doc_red_val=None
    X_doc_red_test=None


    X_doc_red = X_doc[ca_train.index]

    if ca_test is not None:
        X_doc_red_test = X_doc[ca_test.index]
    if ca_val is not None:
        X_doc_red_val = ca_val['normalized_docs'].values


    if svd_comp is not None:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=svd_comp)

        X_doc_red = svd.fit_transform(X_doc_red)
        if ca_val is not None:
            X_doc_red_val = svd.transform(X_doc_red_val)
        if ca_test is not None:
            X_doc_red_test = svd.transform(X_doc_red_test)

    return X_doc_red, X_doc_red_val, X_doc_red_test, svd


def fit_and_svd_structure_only(candidates, X,k=0.8, k2=0.1, svd_comp=50):
    # divide to train and test by repos:
    from collections import Counter
    repocount = []
    import random

    for i, pa in candidates.iterrows():
        p = os.path.normpath(pa['path'])
        repocount.append(p.split(os.sep)[-2] + os.sep + p.split(os.sep)[-1])

    counter = Counter(repocount)
    items = [i for i in counter.keys()]
    random.shuffle(items)

    l= int(k*len(items))
    l2=int(k2*len(items))

    trainmask = candidates['path'].str.contains('|'.join(items[0:l]))
    valmask = candidates['path'].str.contains('|'.join(items[l:l+l2]))
    testmask = candidates['path'].str.contains('|'.join(items[l+l2:]))

    ca_train = candidates[trainmask]
    ca_val = candidates[valmask]
    ca_test = candidates[testmask]
    # Truncated SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=svd_comp)
    print('here')
    print(X[ca_train.index, :].shape)
    X_svd = svd.fit_transform(X[ca_train.index, :])
    print(svd.explained_variance_ratio_.sum())

    X_svd_test = svd.transform(X[ca_test.index, :])
    X_svd_val = svd.transform(X[ca_val.index, :])

    return ca_train, X_svd, ca_val, X_svd_val, ca_test,  X_svd_test, items[0:l],items[l:l+l2],items[l+l2:]


def reduction_to_repos(X, df_data, items):
    from scipy.sparse import csr_matrix
    x = csr_matrix((len(items), X.shape[1]), dtype=X.dtype)
    for i in range(len(items)):
        x_item = X[df_data['path'].str.contains(items[i])]
        # simple reduction
        x_item = np.mean(x_item, axis=0)
        x[i] = x_item
    return x


def reduction_to_repos_svd(X, df_data, items,n=2):
    from scipy.sparse import csr_matrix
    from scipy.sparse import vstack
    x = csr_matrix((len(items), X.shape[1]*n), dtype=X.dtype)
    for i in range(len(items)):
        x_item = X[df_data['path'].str.contains(items[i])]

        # Truncated SVD
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n)
        while x_item.shape[0]<=n:
            x_item=vstack([csr_matrix(x_item), csr_matrix(x_item)])
        x[i] = x_svdt.reshape(1, X.shape[1] * n)

    return x



