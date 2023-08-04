import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
# get tags related to topic by comparing two populations

class TopicTag:
    """ evaluating topic for repository using two paths of repos, a topic path and a standard path.
    - data_tag : Train Matrix n_samples X n_tags
    - data_features : Train Matrix n_samples X n_features
    """
    def __init__(self,name="topic"):
        self.topic_results=[]
        self.topic_struct_results=[]
        self.repos=[]
        self.name=name

    def clarity_tags(self, df_train, X_svd_train, doc_train, labels, ntags=100 ,toptags=10,split_level=1):
        """Get tags related to the topic paths"""
        repos_base = ~labels
        repos_target = labels
        df_train_temp=df_train.copy()
        df_train_temp['path']=df_train_temp['path'].apply(lambda x: x.split(os.sep)[split_level])
        df_train_temp.reset_index(drop=True, inplace=True)
        df_train_temp['index']=df_train_temp.index
        df_train_temp=df_train_temp[repos_target]
        repo_i=df_train_temp.groupby('path').index.apply(list)


        doc_train_target=np.zeros((len(repo_i.index),doc_train.shape[1]))
        for i in range(len(repo_i.index)):
            doc_train_target[i]=np.clip(np.log(np.sum(doc_train[repo_i[i]].toarray(),axis=0)+1),0,1.0)

        df_train_temp=df_train.copy()
        df_train_temp['path']=df_train_temp['path'].apply(lambda x: x.split(os.sep)[split_level])
        df_train_temp.reset_index(drop=True, inplace=True)
        df_train_temp['index']=df_train_temp.index
        df_train_temp=df_train_temp[repos_base]
        repo_i=df_train_temp.groupby('path').index.apply(list)
        doc_train_base=np.zeros((len(repo_i.index),doc_train.shape[1]))
        for i in range(len(repo_i.index)):
            doc_train_base[i]=np.clip(np.log(np.sum(doc_train[repo_i[i]].toarray(),axis=0)+1),0,1.0)

        dist_tags_target = (doc_train_target.sum(axis=0)+1e-4)/(len(repos_target)+1)
        dist_tags_base = (doc_train_base.sum(axis=0)+5+1e-4)/(len(repos_base)+1)#+1 to avoid rare tags.
        norm_dist_base = dist_tags_base.T / np.linalg.norm(dist_tags_base.T)
        norm_dist_target = dist_tags_target.T / np.linalg.norm(dist_tags_target.T)
        clarity_ratio = (norm_dist_target) / (norm_dist_base)
        self.weights_tags = clarity_ratio / np.linalg.norm(clarity_ratio)
        try:
            self.tags_rank = (np.array(self.weights_tags).argsort()[-ntags:])
        except:
            self.tags_rank = (self.weights_tags.T.argsort()[-ntags:])
            return self.tags_rank, self.weights_tags

        self.toptags=toptags
        self.X_svd_top, self.df_top = self.get_top_topic_vectors(df_train[labels], X_svd_train[labels], doc_train[labels], self.tags_rank,ntags=toptags)
        #self.weights_tags=np.array(self.weights_tags)[:, 0]

        return self.tags_rank, self.weights_tags
    
    def clarity_tags_svd(self, doc_train, labels, ntags=100,rare_rm=0):
        """Get tags related to the topic paths"""
        repos_base = ~labels
        repos_target = labels
        dist_tags_target = (doc_train[repos_target].sum(axis=0)+1e-4)/(len(repos_target))
        dist_tags_base = (doc_train[repos_base].sum(axis=0)+rare_rm+1e-4)/(len(repos_base))
        norm_dist_base = dist_tags_base.T / np.linalg.norm(dist_tags_base.T)
        norm_dist_target = dist_tags_target.T / np.linalg.norm(dist_tags_target.T)
        clarity_ratio = (norm_dist_target) / (norm_dist_base)
        self.weights_tags = clarity_ratio / np.linalg.norm(clarity_ratio)
        
        self.tags_rank = (np.array(self.weights_tags.T.argsort())[0][-ntags:])
        return self.tags_rank, self.weights_tags


    # get results from distribution using dot product.
    def get_repo_score(self, repo, cand, docs, tags_rank,i=0):
        """Get scores from the dot product of clarity weights matrix(1Xn) and the tags samples matrix(mXn) """
        # get repo score

        ml_tags_list = np.array(list(tags_rank))
        weight_rank = np.array(self.weights_tags[ml_tags_list[::-1]])
        docs = docs.toarray()
        ml_tags_rank = np.array(list(tags_rank))[::-1]
        xdoc_repo = np.array([docs[i] for i, candidate in zip(range(docs.shape[0]), cand['path'].str.contains(repo))
                              if candidate is True])
        repo_tags = xdoc_repo.sum(axis=0)
        repo_ranked_tags = np.array([repo_tags[tag] for tag in ml_tags_rank])
        score_repo = np.sum(repo_ranked_tags * weight_rank)
        return score_repo / xdoc_repo.shape[0]

        # get results from distribution using dot product.
    def get_repo_score_X(self, repo, cand, X_test, tags_rank,weight_rank,i=0):
        """Get scores from the dot product of clarity weights matrix(1Xn) and the tags samples matrix(mXn) """
        if cand is not None:
            #xdoc_repo = X_test[cand['path'].str.contains(repo)]
            xdoc_repo = X_test[np.array(cand['path'].str.contains(repo)).nonzero()[0]]
            repo_tags=xdoc_repo
            if xdoc_repo.shape[0]==0:
                return -1.0
        else:
            repo_tags=X_test[i]
        repo_ranked_tags = repo_tags.T[tags_rank] # check transpose valid

        if cand is not None:
            repo_ranked_tags = (repo_ranked_tags.mean(axis=1))

        score_repo = np.sum(repo_ranked_tags.T * weight_rank)/(np.linalg.norm(repo_ranked_tags)*np.linalg.norm(weight_rank)+1e-20)

        return score_repo

    def calculate_similarity_to_clarity(self,candidates, items, X,offset=True):
        scores = []
        # get repo score
        tags_rank = np.array(list(self.tags_rank))[::-1]
        weight_rank = self.weights_tags[tags_rank]
        if offset:
            weight_rank -= np.mean(weight_rank)

        for i in tqdm(range(len(items)), desc="Computing Likelihood..."):
            similarity_score = self.get_repo_score_X(items[i], candidates, X, tags_rank,weight_rank,i)

            scores.append(similarity_score)
        return scores

    def get_ranked_tags_in_repo(self,candidates, items, X, N_tags=10):
        scores = []

        for item in tqdm(items, desc="Computing Likelihood..."):
            xdoc_repo=X[candidates['path'].isin([item])]
            if xdoc_repo.shape[0] == 0:
                scores.append([-1]*N_tags)
                continue
            ranked_list=np.array((xdoc_repo).mean(axis=0))[0].argsort()[::-1][0:N_tags]
            scores.append(ranked_list)
        return scores


    def get_top_topic_vectors(self, candidates, X_svd, X_doco, tags_rank, ntags=10):
        """provide the most relevant svd vectors which are associated with the top ranked tags"""
        tags_list = np.array(list(tags_rank))[-ntags:-1]
        mask = []
        for d in X_doco:
            t = np.array(d.todense()).nonzero()[1]
            mask.append(any(x in t for x in tags_list))
        # tags_res=np.array(X_doc_test.todense()).argsort()[:,-6:]
        tags_i = np.where(mask)[0]
        return X_svd[tags_i], candidates.iloc[tags_i]

    def get_snippet_based_score(self, repo, candidates_top_test, X_svd_top_test, kernel):
        likelihood = 0
        xin = X_svd_top_test[candidates_top_test['path'].str.contains(repo)]
        for id in range(len(xin)):
            likelihoodid = kernel.compute_likelihood(self.X_svd_top, [xin[id]])  # Likelyhood using a Gaussian kernel
            if np.mean(likelihoodid) > 0.5e-3: # hard threshold to count relevant close snippet in structure.
                likelihood += np.mean(likelihoodid)
        return likelihood / (len(xin) + 1e-10)

    def score_recalls(self, candidates_df, X_svd_test, docs, labels,items, kernel, threshold=None):
        # Calculate the tag score and structure score for all test repos, and evaluate results using a threshold.
        X_svd_top_test, candidates_top_test = self.get_top_topic_vectors(candidates_df, X_svd_test, docs, self.tags_rank,ntags=self.toptags)

        if threshold!=None:
            self.threshold=threshold
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        repocount = []

        self.true_labels=[]
        self.topic_results=[]
        self.topic_struct_results=[]
        for i in range(len(items)):

            label_topic =labels[list(candidates_df['path'].str.contains(items[i]))][0] # repo level label

            #label_topic = labels[i]

            self.true_labels.append(label_topic)
            score = self.get_repo_score(items[i], candidates_df, docs, self.tags_rank)
            self.topic_results.append(score)
            score_struct = self.get_snippet_based_score(items[i], candidates_top_test, X_svd_top_test, kernel)
            self.topic_struct_results.append(score_struct)
            s = 0.5*score_struct+0.5*score
            if label_topic: # TO DO: remove and use only 1D density.
                if s > self.threshold:
                    tp += 1
                if s < self.threshold:
                    fn += 1
            else:
                if s > self.threshold:
                    fp += 1
                if s < self.threshold:
                    tn += 1
        print(tp,tn,fp,fn)
        print('Precision:', tp / (tp + fp+1e-10))
        print('Positive Recalls', tp / (tp + fn+1e-10), 'Negative recalls', tn / (tn + fp+1e-10))

        self.repos=items

        return 0.5 * tp / (tp + fn+1e-10) + 0.5 * tn / (tn + fp+1e-10), self.topic_results, self.topic_struct_results

    def get_simlarities_docstrings(self, candidates_df, docs, items):
        topic_results=[]
        for i in range(len(items)):
            score = self.get_repo_score(items[i], candidates_df, docs, self.tags_rank)
            topic_results.append(score)
        return topic_results

    def aggregate_to_repo_docs(self,docs, df):
        sorted_docs=[]
        repocount = []
        for i, pa in df.iterrows():
            p = os.path.normpath(pa['path'])
            repocount.append(p.split(os.sep)[5] + os.sep + p.split(os.sep)[6])
        counter = Counter(repocount)
        repos = [i for i in counter.keys()]
        for repo in repos:
            xdoc_repo = docs[df['path'].str.contains(repo)]
            repo_doc = (xdoc_repo.sum(axis=0))
            sorted_docs.append(np.array(repo_doc)[0])
        return sorted_docs, repos

    def get_density_scores(self):
        # Normalise tag score and structure score results to get densities/probability for topic.
        std_struct_results = np.array(self.topic_struct_results) / max(self.topic_struct_results)
        std_results = np.array(self.topic_results) / max(self.topic_results)
        density_scores = np.sqrt(std_results**2+std_struct_results**2)
        return density_scores

    def optimise_score(self, candidates_df, X_svd_test, docs, topic_str, kernel, init_guess=0.01):
        import scipy
        self.threshold = scipy.optimize.fmin(lambda x: -self.score_recalls(candidates_df,X_svd_test, docs, topic_str, kernel, x), init_guess)




