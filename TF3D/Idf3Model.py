import numpy as np
import pandas as pd
import pickle
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import importlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score
from scipy.sparse import coo_matrix, vstack
# internal  Code
from TF3D import Data_Process
import senatus_code.feature_extraction.vectorize as vectorize
from senatus_code.feature_extraction.featurizer import TreeRelationFeaturizer
from TF3D import topic_metric as topic
from TF3D import dependancies_processing as deps


class Tf3D:
    def __init__(self, data_path, topics, items_train=None, items_val=[], items_test=[]):
        """Embedding and evaluating object for term frequency model of topic tagging.
        :param data_path: Path for dataset of repos.
        :param topics: Topic list for tagging.
        :param items_train: Repo list of names to train.
        :param items_val: Repo list of names to validate.
        :param items_test: Repo list of names to test.
        """
        print("Tf3D for tagging topics")
        self.data_path=data_path
        self.items_train=items_train
        self.items_val=items_val
        self.items_test=items_test

        dataset_description = pd.read_csv(os.path.join(data_path,'results_csv.csv'))
        self.dataset_description = dataset_description
        self.topics = topics
        if items_train!=None:
            # vectorize items:
            items_train_path=[data_path+os.sep+it for it in items_train]
            items_val_path=[data_path+os.sep+it for it in items_val]
            items_test_path=[data_path+os.sep+it for it in items_test]
            vectorizer=vectorize.CountVectorizer(TreeRelationFeaturizer(),write2disk=True,binary=True)
            self.X = vectorizer.fit_transform(items_train_path, 'python')

            self.X_val_test = vectorizer.transform(items_val_path + items_test_path, 'python')
            vectorizer.serialize_raw_features(os.path.join(data_path,'raw_features.msgpack'))

            self.X=vstack([self.X,self.X_val_test]).tocsr()
            print('structure size:',self.X.shape)

            # prepare data docstring
            data_red = Data_Process.prepare_data_new(vectorizer.features_raw, data_path)
            self.data_red=data_red


            self.ca_train, self.ca_val, self.ca_test=Data_Process.divide_data_according_to_items(data_red, dataset_description,items_train,items_val,items_test,IdentiferLevel=1)
            data_red.head()
        else:
            print('vectorizing all data path...')
            vectorizer=vectorize.CountVectorizer(TreeRelationFeaturizer(),write2disk=True,binary=True)
            self.X = vectorizer.fit_transform(data_path, 'python')
             # prepare data docstring
            data_red = Data_Process.prepare_data_new(vectorizer.features_raw, data_path)
            self.ca_train, self.ca_val, self.ca_test, self.items_train, self.items_val, self.items_test=Data_Process.get_training_val_test_items(data_red, dataset_description, k=0.6,k2=0.2, IdentiferLevel=5,size=10)
            items_train=self.items_train
            items_val=self.items_val
            items_test=self.items_test

        self.X_dep_train, self.X_dep_val, self.X_dep_test = deps.vectroize_dependancies(data_path, self.items_train, self.items_val, self.items_test)
        self.X_svd_train, self.X_svd_val,  self.X_svd_test = Data_Process.fit_and_svd_features(self.X, self.ca_train, self.ca_val, self.ca_test, svd_comp=100)
        self.doc_train, self.doc_val,  self.doc_test, self.vectorizer_ =Data_Process.fit_docs(self.ca_train, self.ca_val, self.ca_test)

        # Reduction by average for REPO level, only relevant if using SVD but not doing that anymore (to be removed)
        x_train = Data_Process.reduction_to_repos(self.X_svd_train, self.ca_train, self.items_train)
        x_val = Data_Process.reduction_to_repos(self.X_svd_val, self.ca_val,self.items_val)
        x_test = Data_Process.reduction_to_repos(self.X_svd_test, self.ca_test, self.items_test)

        # get topics for classifications
        le = preprocessing.MultiLabelBinarizer()
        topics_labels=[dataset_description[dataset_description['full_name'] == it]['featured_topics'].tolist() for it in items_train]
        topics_labels_val=[dataset_description[dataset_description['full_name'] == it]['featured_topics'].tolist() for it in items_val]
        topics_labels_test=[dataset_description[dataset_description['full_name'] == it]['featured_topics'].tolist() for it in items_test]
        topics_labels=[[l for l in ast.literal_eval(labels[0]) if l in self.topics] for labels in topics_labels]
        topics_labels_val=[[l for l in ast.literal_eval(labels[0]) if l in self.topics] for labels in topics_labels_val]
        topics_labels_test=[[l for l in ast.literal_eval(labels[0]) if l in self.topics] for labels in topics_labels_test]

        le.fit(topics_labels)
        self.le_train=le.transform(topics_labels)
        self.le_val=le.transform(topics_labels_val)
        self.le_test=le.transform(topics_labels_test)
        self.topics=list(le.classes_)
        print(self.topics)

    def get_vocabulary(self, number_keywords) -> dict:
        """Receive top vocabulary from the topics.
        :param number_keywords: int as a no. of top terms.
        :return: list of top terms in vocabulary
        """
        vocabulary = return_top_keywords(self.ca_train, self.X_svd_train, self.doc_train, self.vectorizer_, self.topics, number_keywords)
        return vocabulary

    def calc_embedding(self):
        """Calculate embedding using TF model and cosine similarity with embedding saved in objcet.
        :return: list of embedding in 2D matrix (N_repos) X (3XTopics) for training, val, and test sets.
        """
        score_train, score_val, score_test = eval_topic_docstrings(self.ca_train, self.ca_val, self.ca_test,self.X_svd_train, self.doc_train, self.doc_val,  self.doc_test, self.items_train, self.items_val, self.items_test, self.vectorizer_, self.topics)
        scoredep_train, scoredep_val, scoredep_test = eval_topic_deps(self.X_dep_train, self.X_dep_val, self.X_dep_test, self.X, self.items_train, self.items_val, self.items_test, self.topics,self.dataset_description)
        scoreast_train, scoreast_val, scoreast_test= eval_topic_ast(self.ca_train, self.ca_val, self.ca_test, self.X, self.items_train, self.items_val, self.items_test, self.topics)
        """plot_topics_labels(score_train,score_val,score_test,
                               scoreast_train,scoreast_val,scoreast_test,
                         scoredep_train,scoredep_val,scoredep_test,self.le_train,self.le_val,self.le_test,self.topics)"""
        try:
            topics=self.topics
            # rearrange to (N_topicsX3) X N_items
            self.x_tr=[]
            for i in range(len(self.items_train)):
                x_tri=[score_train[t][i] for t in topics]
                x_tri.extend([scoreast_train[t][i] for t in topics])
                x_tri.extend([scoredep_train[t][i] for t in topics])
                self.x_tr.append(np.array(x_tri))

            self.x_va=[]
            for i in range(len(self.items_val)):
                x_tri=[score_val[t][i] for t in topics]
                x_tri.extend([scoreast_val[t][i] for t in topics])
                x_tri.extend([scoredep_val[t][i] for t in topics])
                self.x_va.append(np.array(x_tri))

            self.x_te=[]
            for i in range(len(self.items_test)):
                x_tei=[score_test[t][i] for t in topics]
                x_tei.extend([scoreast_test[t][i] for t in topics])
                x_tei.extend([scoredep_test[t][i] for t in topics])
                self.x_te.append(np.array(x_tei))
            return self.x_tr, self.x_va, self.x_te

        except:
            print('Failed but saved!')
            return score_train, score_val, score_test, scoredep_train, scoredep_val, scoredep_test, scoreast_train, scoreast_val, scoreast_test

    def classify_sklearn_test_rf(self,maxf=10):
        """Simple classification using random-forest on embeddings"""
        from sklearn.ensemble import RandomForestClassifier
        neigh = RandomForestClassifier(max_features=maxf,random_state=42)
        neigh.fit(np.array(self.x_tr),  list(self.le_train))

        y_train_sim =neigh.predict(np.array(self.x_tr))
        if len(self.x_va)>0:
            y_val_sim=neigh.predict(np.array(self.x_va))
        y_test_sim=neigh.predict(np.array(self.x_te))

        from sklearn.metrics import precision_score
        from sklearn.metrics import f1_score

        precision=precision_score(self.le_test, y_test_sim, average='micro')
        f1 = f1_score(self.le_test, y_test_sim, average='micro')
        print('precision:',precision,'f1: ',f1)
        print(y_test_sim)
        print(self.le_test)
        if len(self.x_va)>0:
            precision=precision_score(self.le_val, y_val_sim, average='micro')
            f1 = f1_score(self.le_val, y_val_sim, average='micro')
            print('precision:',precision,'f1: ',f1)
            print(y_val_sim)
            print(self.le_val)


def eval_topic_docstrings(ca_train, ca_val, ca_test, X_svd_train,
                          doc_train, doc_val,doc_test, items_train,
                          items_val, items_test, vectorizer_, topics):
    """Generate embedding for docstrings given a dataframes processed and list of training, val, test sets"""
    cltopic={}
    score_train={}
    score_val={}
    score_test={}
    for t in tqdm(topics):
        print(t)
        cltopic[t]=topic.TopicTag()
        labels_train=np.array([True if t in labels else False for labels in ca_train['featured_topics'].tolist()])
        tags_vocab=vectorizer_.get_feature_names()

        tags_rank, weights_tags = cltopic[t].clarity_tags(ca_train,X_svd_train,doc_train, labels_train,ntags=2500,toptags=5)

        print([tags_vocab[x] for x in tags_rank][-20:-1])
        plt.plot(weights_tags-np.mean(weights_tags))

        score_train[t]=cltopic[t].get_simlarities_docstrings(ca_train,doc_train,items_train)
        score_val[t]=cltopic[t].get_simlarities_docstrings(ca_val,doc_val,items_val)
        score_test[t]=cltopic[t].get_simlarities_docstrings(ca_test,doc_test,items_test)

    return score_train, score_val, score_test

def return_top_keywords(ca_train, X_svd_train, doc_train, vectorizer_, topics, number_keywords=200):
    """Generate embedding for docstrings given a dataframes processed and list of training, val, test sets"""
    cltopic={}
    vocabulary = {}
    for t in tqdm(topics):
        cltopic[t]=topic.TopicTag()
        labels_train=np.array([True if t in labels else False for labels in ca_train['featured_topics'].tolist()])
        tags_rank, weights_tags = cltopic[t].clarity_tags(ca_train,X_svd_train,doc_train, labels_train,ntags=2500,toptags=5)
        tags_vocab=vectorizer_.get_feature_names()
        vocabulary[t] = [tags_vocab[x] for x in tags_rank[-number_keywords+1:-1]]
    return vocabulary

def eval_topic_ast(ca_train, ca_val, ca_test, X,items_train, items_val, items_test, topics):
    asttopic={}
    scoreast_train={};scoreast_val={};scoreast_test={}
    for t in topics:
        asttopic[t]=topic.TopicTag()
        labels_train=np.array([True if t in labels else False for labels in ca_train['featured_topics'].tolist()])
        # generate clarity distribution for each topic class using training.
        tags_rank, weights_tags = asttopic[t].clarity_tags_svd( X[ca_train.index], labels_train, ntags=00,rare_rm=5)
        plt.plot(weights_tags-np.mean(weights_tags))

        scoreast_train[t]=asttopic[t].calculate_similarity_to_clarity(ca_train,items_train,X[ca_train.index],offset=False)
        scoreast_val[t]=asttopic[t].calculate_similarity_to_clarity(ca_val,items_val,X[ca_val.index],offset=False)
        scoreast_test[t]=asttopic[t].calculate_similarity_to_clarity(ca_test,items_test,X[ca_test.index],offset=False)
    return scoreast_train, scoreast_val, scoreast_test

def eval_topic_deps(X_dep_train, X_dep_val, X_dep_test, X,items_train, items_val, items_test,topics, dataset_description):
    deptopic={}
    scoredep_train={};scoredep_val={};scoredep_test={}
    for t in topics:
        deptopic[t]=topic.TopicTag()

        topics_labels=[dataset_description[dataset_description['full_name'] == it]['featured_topics'].iloc[0] for it in items_train]
        labels_train=np.array([True if t in labels else False for labels in topics_labels]) ## Change == to inclusion

        # generate clarity distribution for each topic class using training.
        tags_rank, weights_tags = deptopic[t].clarity_tags_svd(X_dep_train, labels_train, ntags=200,rare_rm=1)
        plt.plot(weights_tags-np.mean(weights_tags))

        scoredep_train[t]=deptopic[t].calculate_similarity_to_clarity(None,items_train,X_dep_train.toarray(),offset=False)
        scoredep_val[t]=deptopic[t].calculate_similarity_to_clarity(None, items_val,X_dep_val.toarray(),offset=False)
        scoredep_test[t]=deptopic[t].calculate_similarity_to_clarity(None,items_test,X_dep_test.toarray(),offset=False)
    return scoredep_train, scoredep_val, scoredep_test

def plot_topics_labels(score_train,score_val,score_test,
                       scoreast_train,scoreast_val,scoreast_test,
                       scoredep_train,scoredep_val,scoredep_test, labels_train, labels_val, labels_test,topics):
    for t in topics:
        fig=plt.figure(t)

        ax=fig.add_subplot(projection='3d')
        sep=1e-3
        x0b=((np.clip(scoreast_train[t],a_min=0,a_max=None)+sep))#/np.max(topic_sim_train))
        x1b=((np.clip(score_train[t],a_min=0,a_max=None)+sep))#/np.max(scores_train_all))
        x2b=((np.clip(scoredep_train[t],a_min=0,a_max=None)+sep))#/np.max(scores_train_all))


        x0c=((np.clip(scoreast_test[t],a_min=0,a_max=None)+sep))#/np.max(topic_sim_val))
        x1c=((np.clip(score_test[t],a_min=0,a_max=None)+sep))#/np.max(scores_val_all))
        x2c=((np.clip(scoredep_test[t],a_min=0,a_max=None)+sep))#/np.max(scores_val_all))


        #ax.scatter(np.log(x0),np.log(x1),np.log(x2),c=labels_val)
        ax.scatter(np.log(x0c),np.log(x1c),np.log(x2c),c=labels_test)
        ax.scatter(np.log(x0b),np.log(x1b),np.log(x2b),c=labels_train)
        #plt.scatter(np.log(x0b),np.log(x1b),np.log(x2b),c=labels_knn_train)
        ax.set_title(t)
        ax.set_xlabel('Tags')
        ax.set_ylabel('Structure')
        ax.set_zlabel('Dependancies')
