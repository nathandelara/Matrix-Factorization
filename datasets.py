# for datasets import
import urllib.request
import os
import tarfile
import glob
import shutil
import signal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def import_graph_df(dataset, url="http://konect.uni-koblenz.de/downloads/tsv/", compression="bz2"):
    dataset_filename = dataset + ".tar." + compression
    download = urllib.request.urlretrieve(url + dataset_filename, dataset_filename)
    tf = tarfile.open(dataset_filename, "r:" + compression) 
    tf.extractall()
    os.chdir(dataset)
    for filename in glob.glob('out.*'):
        f = open(filename)
        line = f.readline()
        graph_type = line.split(' ')[2][:-1]
        graph_df = pd.read_table(f,sep = '\s+',names = ['source','target','weight','time'],comment='%')    
        f.close()
    tf.close()
    os.chdir('..')  
    os.remove(dataset_filename)
    shutil.rmtree(dataset)
    return graph_df, graph_type


def biblock_adj(adj_matrix):
    return sparse.bmat([[None, adj_matrix], [adj_matrix.T, None]], format='csr')


class BDataset:
    """
    A class for Bipartite graphs.
    """
    def __init__(self, name):        
        # get the graph
        self.name = name
        
        self.df, self.type = import_graph_df(name)
        row, col, data = self.df['source'].values, self.df['target'].values, np.ones(len(self.df), dtype=bool)     
             
        n1, n2 = max(row), max(col)
        self.n_nodes = (n1, n2)
        self.adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(n1+1, n2+1))[1:,1:]
        
        # extraction of the largest connected component
        biblock_matrix = biblock_adj(self.adj_matrix)
        n_cc, cc_labels = connected_components(biblock_matrix, directed=False)
        labels, labels_counts = np.unique(cc_labels, return_counts=True)
        max_cc_label = labels[np.argmax(labels_counts)]
        max_cc_indices = np.where(cc_labels == max_cc_label)[0]
        split_idx = np.searchsorted(max_cc_indices, n1)
        idx_V1, idx_V2 = max_cc_indices[:split_idx], max_cc_indices[split_idx:] - n1
        self.adj_matrix = self.adj_matrix[idx_V1, :]
        self.adj_matrix = (self.adj_matrix.tocsc()[:, idx_V2]).tocsr()
        
        # edges are counted without multiplicity
        self.n_edges = self.adj_matrix.nnz
        self.n_nodes = self.adj_matrix.shape
 
    def display(self):
        print(self.name+": {} nodes, {:d} edges.".format(self.n_nodes, self.n_edges))
        

class Dataset:
    """
    A class for Undirected and Directed graphs.
    """
    def __init__(self, name, directed=False):        
        # get the graph
        self.name = name
        self.directed = directed
        
        self.df, self.type = import_graph_df(name)
        row, col, data = self.df['source'].values, self.df['target'].values, np.ones(len(self.df), dtype=bool)     
             
        self.n_nodes = max(max(row), max(col))
        self.adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(self.n_nodes+1, self.n_nodes+1))[1:,1:]
        if not directed:
            self.adj_matrix = self.adj_matrix.maximum(self.adj_matrix.T)
        
        # extraction of the largest connected component
        if directed:
            n_cc, cc_labels = connected_components(self.adj_matrix, directed=True)
        else:
            n_cc, cc_labels = connected_components(self.adj_matrix, directed=False)
        labels, labels_counts = np.unique(cc_labels, return_counts=True)
        max_cc_label = labels[np.argmax(labels_counts)]
        max_cc_indices = np.where(cc_labels == max_cc_label)[0]
        self.adj_matrix = self.adj_matrix[max_cc_indices, :]
        self.adj_matrix = (self.adj_matrix.tocsc()[:, max_cc_indices]).tocsr()
        
        # edges are counted with multiplicity
        self.n_edges = self.adj_matrix.nnz
        self.n_nodes = self.adj_matrix.shape[0]
 
    def display(self):
        print(self.name+": {} nodes, {:d} edges.".format(self.n_nodes, self.n_edges))
