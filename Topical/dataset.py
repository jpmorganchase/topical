import warnings
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import numpy as np
import torch
from typing import Union, List

warnings.filterwarnings('ignore')

class Dataset(torch.utils.data.Dataset):
    def __init__(self,labels, embeddings, le: MultiLabelBinarizer, repo_size=15, cca: Union[CCA, None] = None, pca: Union[PCA, None] = None, prepadding: bool=True):
        self.labels = le.transform(labels)
        self.pca = pca
        self.cca = cca
        self.repo_size = repo_size
        self.prepadding = prepadding
        if cca is not None:
            embeddings = self.transform_cca(embeddings)
        elif pca is not None:
            embeddings = self.transform_pca(embeddings)
        else:
            pass
        self.repo_size = repo_size
        self.embeddings = [self.reshape_embedding(torch.tensor(embedding).float())
                           for embedding in embeddings]
        self.le=le

    def __len__(self):
        return len(self.labels)

    def reshape_embedding(self, selected_embeddings):
        attention_mask = torch.ones(self.repo_size)
        if selected_embeddings.shape[0]<self.repo_size and self.prepadding is True:
            final_embeddings = torch.zeros((self.repo_size, selected_embeddings.shape[1]))
            final_embeddings[len(final_embeddings)-selected_embeddings.shape[0]:] = selected_embeddings
            attention_mask[:len(final_embeddings)-selected_embeddings.shape[0]] = torch.zeros(self.repo_size-selected_embeddings.shape[0])
        elif selected_embeddings.shape[0]<self.repo_size:
            final_embeddings = torch.zeros((self.repo_size, selected_embeddings.shape[1]))
            final_embeddings[:selected_embeddings.shape[0]] = selected_embeddings
            attention_mask[selected_embeddings.shape[0]:] = torch.zeros(self.repo_size-selected_embeddings.shape[0])
        elif selected_embeddings.shape[0]>self.repo_size:
            final_embeddings = selected_embeddings[:self.repo_size]
        else:
            final_embeddings = selected_embeddings
        return final_embeddings, attention_mask

    def __getitem__(self, idx):
        selected_embeddings = self.embeddings[idx]
        item = {'embeddings': selected_embeddings[0]}
        item['labels'] = torch.tensor(self.labels[idx])
        item['attention_mask'] = selected_embeddings[1]
        return item

    def transform_pca(self, embeddings) -> List[torch.tensor]:
        embeddings = [self.pca.transform(emb) for emb in embeddings]
        return embeddings

    def transform_cca(self, embeddings) -> List[torch.tensor]:
        ast_embeddings = [[emb[:768] for emb in embs] for embs in embeddings]
        code_embeddings = [[emb[768:] for emb in embs] for embs in embeddings]
        embeddings = [np.concatenate(self.cca.transform(emb1, emb2), axis=1) for emb1, emb2 in zip(ast_embeddings, code_embeddings)]
        return embeddings

def fit_cca_components(embeddings, number_components: int) -> CCA:
    cca = CCA(n_components=number_components)
    embeddings_ast = np.array([t[:768] for emb in embeddings for t in emb])
    embeddings_code = np.array([t[768:1536] for emb in embeddings for t in emb])
    #embeddings_doc = np.array([t[1536:] for emb in embeddings for t in emb])
    cca.fit(embeddings_ast, embeddings_code)#, embeddings_doc)
    return cca

def fit_label_binarizer(labels) -> MultiLabelBinarizer:
    #labels = np.array([l for label in labels for l in label])
    le = MultiLabelBinarizer()
    le.fit_transform(labels)
    return le

def fit_pca_components(embeddings, number_components: int) -> PCA:
    pca = PCA(n_components=number_components)
    embeddings = np.array([t for emb in embeddings for t in emb])
    pca.fit(embeddings)
    return pca
