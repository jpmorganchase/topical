from tqdm import tqdm
import os
import json
import re
import ast
from sklearn.feature_extraction.text import CountVectorizer


def vectroize_dependancies(data_path, items_train, items_val, items_test):
    depslist=[]
    
    for i in tqdm(items_train):
        repos_path=data_path+os.sep+i
        dfl=populate_simple_import(repos_path)
        depslist.append(dfl)
    depslist2=[]
    for i in tqdm(items_val):
        repos_path=data_path+os.sep+i
        dfl=populate_simple_import(repos_path)
        depslist2.append(dfl)
    depslist3=[]
    for i in tqdm(items_test):
        repos_path=data_path+os.sep+i
        dfl=populate_simple_import(repos_path)
        depslist3.append(dfl)

    vectdep=CountVectorizer([],binary=True,ngram_range=(1,1))
    X_dep_train=vectdep.fit_transform(depslist)
    X_dep_val=vectdep.transform(depslist2)
    X_dep_test=vectdep.transform(depslist3)
    return X_dep_train, X_dep_val, X_dep_test


def populate_simple_import(repo):
    modules=set()
    def visit_Import(node):
        for name in node.names:
            modules.add(name.name.split(".")[0])
    def visit_ImportFrom(node):
        # if node.module is missing it's a "from . import ..." statement
        # if level > 0 it's a "from .submodule import ..." statement
        if node.module is not None and node.level == 0:
            modules.add(node.module.split(".")[0])

    node_iter = ast.NodeVisitor()
    node_iter.visit_Import = visit_Import
    node_iter.visit_ImportFrom = visit_ImportFrom
    listdep=[]
    files,_=walk_py(repo)
    for file in files:
        modules=set()
        try:
            with open(file) as f:
                node_iter.visit(ast.parse(f.read())) 
                listdep.extend(list(modules))             
        except:
            continue;
    return ' '.join(listdep)

def walk_py(repo_path, lang_ex=".py"):
    py_paths=[]
    repo_paths=[]
    for subdir, dirs, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith(lang_ex): 
                py_paths.append(os.path.join(subdir, filename))
                repo_paths.append(os.path.join(subdir))
    return py_paths,repo_paths

def get_pycg(repo, repo_file):
    json_path=repo+'/cg.json'
    
#     path="/Users/V773117/datasetshort/allenai_allennlp"
#     os.system('pycg '+'--package '+path +' '+r"$(find allenai_allennlp -type f -name \"*.py\")"+ ' -o '+json_path+' --max-iter 2')
    os.system('pycg '+'--package '+repo +' '+repo_file+' -o '+json_path+' --max-iter 2')
    f = open(json_path)
    data = json.load(f)
    f.close()
    os.remove(json_path)
    return data

def get_two_initials(s): # add fuzzy match?
    if '.' in s:
        return str(s.split('.')[0])+' '+str(s.split('.')[1])
    else:
        return s
    
# populate parent - child edges in a df tokenizd-ready.
def iter_json(data_d,gen=0,parent=[]):
    global df
    if isinstance(data_d,dict):
        keys=list(data_d.keys())
        for k in keys:
            #print(parent,k,gen)
            parent=get_two_initials(parent)
            k0=get_two_initials(k)
            if len(parent)>0:
                df.loc[len(df)]=[parent,k0]
            iter_json(data_d[k],gen+1,k)        
    if isinstance(data_d,list):
        for l in data_d:
            iter_json(l,gen,parent)
    if isinstance(data_d,str):
        parent=get_two_initials(parent)
        data_d=get_two_initials(data_d)
        if len(parent)>0:
            df.loc[len(df)]=[parent,data_d]
        #print(parent,data_d,gen)
        
def populate_repo_df_edges(repo,df):
    files, repos=walk_py(repo)
    for file,rep in zip(files,repos):
        jsontree=get_pycg(repo, file)
        try:
            
            iter_json(jsontree)
        except:
            continue;
    df=df.drop_duplicates()
    return df