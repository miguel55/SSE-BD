#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:29:44 2020

@author: mmolina
"""


import numpy as np
import os
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas
import matplotlib.pyplot as plt
from config import cfg
import umap
import scipy.io as sio

sns.set(rc={'figure.figsize':(11.7,8.27)})


# 1. GMM in the original space 
# Read and organize the data
#NxTxF
data = sio.loadmat(os.path.join(cfg.data_dir,'cnic_dataset.mat'))['features_lstm']
ids=data[:,:,cfg.n_features:]
data=data[:,:,:cfg.n_features]
train_data=data[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.train_groups)),int((data.shape[1]-1)/2),:].copy().squeeze()
test_data=data[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.test_groups)),int((data.shape[1]-1)/2),:].copy().squeeze()

ids_train=ids[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.train_groups)),int((data.shape[1]-1)/2),:].copy()
ids_test=ids[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.test_groups)),int((data.shape[1]-1)/2),:].copy()

# Normalize the samples
mean=sio.loadmat(os.path.join(cfg.data_dir,'mean_features.mat'))['mean_lstm']
std=sio.loadmat(os.path.join(cfg.data_dir,'std_features.mat'))['std_lstm']

train_data_norm=(train_data-np.repeat(mean,train_data.shape[0],axis=0))/np.repeat(std,train_data.shape[0],axis=0)
test_data_norm=(test_data-np.repeat(mean,test_data.shape[0],axis=0))/np.repeat(std,test_data.shape[0],axis=0)

# GMM
cluster_agg=GaussianMixture(n_components=cfg.K_selected,n_init=10,random_state=cfg.SEED)
cluster_agg.fit(train_data_norm)
cidx_train=cluster_agg.predict(train_data_norm)
cidx_test=cluster_agg.predict(test_data_norm)


# Obtain performance measurements (entropy, log-likelihood, BIC) for train samples
LE_orig_test=cluster_agg.score_samples(test_data_norm)
score_orig_test=cluster_agg.score(test_data_norm)
posterior=cluster_agg.predict_proba(test_data_norm)
entropy_orig_test=np.mean(-np.sum(posterior*np.log2(posterior+np.finfo(float).eps),axis=1))
bic_test=cluster_agg.bic(test_data_norm)

# 2. GMM en el espacio del embedding para la detección del número de comportamientos
log_likelihood=np.zeros((len(cfg.outputdims),),dtype='float32')
log_likelihood_orig=np.zeros((len(cfg.outputdims),),dtype='float32')
entropies=np.zeros((len(cfg.outputdims),),dtype='float32')
entropies_orig=np.zeros((len(cfg.outputdims),),dtype='float32')
BIC=np.zeros((len(cfg.outputdims),),dtype='float32')
BIC_orig=np.zeros((len(cfg.outputdims),),dtype='float32')
for id_out,out in enumerate(cfg.outputdims):
    model_path=cfg.arch+'_g_{}_gtraj_{}_lr_{}_bs_{}_pnum_{}_traj_{}_out_{}'.format(cfg.beta, cfg.beta_traj, cfg.lr, cfg.batch_size,cfg.pnum, cfg.pnum_traj, out)
    data=sio.loadmat(os.path.join(cfg.train_dir, model_path,'embedding_best.mat'))['embedding'].T
    group=data[:,-1].astype('int')
    data=data[:,:-1]
    n_groups=group.max()
    group_ids=['Group '+str(int(i+1)) for i in range(group.max())]
        
    train_data=data[np.in1d(group,np.array(cfg.train_groups)),:].copy()
    test_data=data[np.in1d(group,np.array(cfg.test_groups)),:].copy()
    
    # Normalize the samples
    mean=np.mean(data,axis=0)[None,:]
    std=np.std(data,axis=0)[None,:]
    train_data_norm=(train_data-np.repeat(mean,train_data.shape[0],axis=0))/np.repeat(std,train_data.shape[0],axis=0)
    test_data_norm=(test_data-np.repeat(mean,test_data.shape[0],axis=0))/np.repeat(std,test_data.shape[0],axis=0)
    
    # GMM for the embedding
    cluster_agg=GaussianMixture(n_components=cfg.K_selected,n_init=10,random_state=cfg.SEED)
    cluster_agg.fit(train_data_norm)
    cidx_train=cluster_agg.predict(train_data_norm)
    cidx_test=cluster_agg.predict(test_data_norm)

    # Obtain performance measurements (entropy, log-likelihood, BIC) for train samples
    LE_embed_test=cluster_agg.score_samples(test_data_norm)
    log_likelihood[id_out]=cluster_agg.score(test_data_norm)
    posterior=cluster_agg.predict_proba(test_data_norm)
    entropies[id_out]=np.mean(-np.sum(posterior*np.log2(posterior+np.finfo(float).eps),axis=1))
    BIC[id_out]=cluster_agg.bic(test_data_norm)

    log_likelihood_orig[id_out]=score_orig_test
    entropies_orig[id_out]=entropy_orig_test
    BIC_orig[id_out]=bic_test
            
# Visualization
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dashed', 'dashed'),    # Same as '--'
     ('solid', 'solid')      # Same as (0, ()) or '-'

]   
               
if not os.path.exists(os.path.join(cfg.result_dir,'evaluation')):
    os.makedirs(os.path.join(cfg.result_dir,'evaluation'))
     
xticks=cfg.outputdims.copy()
xticks.insert(cfg.index_insert,21)

# Log likelihood 
plt.plot(np.arange(0,len(cfg.outputdims)),log_likelihood_orig, label='Original data',linewidth=3,linestyle='dashed')
plt.plot(cfg.index_insert-1+5/16,log_likelihood_orig[0], marker="o", markersize=16, markeredgecolor="tab:blue", markerfacecolor="tab:blue")
plt.plot(np.arange(0,len(cfg.outputdims)),log_likelihood, linestyle='solid', linewidth=3, label='Embedded data')
plt.xticks(np.insert(np.arange(0,int(len(cfg.outputdims))).astype('float'),cfg.index_insert,cfg.index_insert-1+5/16),xticks,fontsize=cfg.TITLE_SIZE-4)
plt.yticks(fontsize=cfg.TITLE_SIZE-2)
plt.xlabel('Dimension of the embedding',fontsize=cfg.TITLE_SIZE)
plt.ylabel('Log-likelihood',fontsize=cfg.TITLE_SIZE)
plt.title('Log-likelihood of test data',fontsize=cfg.TITLE_SIZE)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(cfg.result_dir,'evaluation','log_likelihood.png'),dpi=300)
plt.show()

# Entropies
plt.plot(np.arange(0,len(cfg.outputdims)),entropies_orig, label='Original data',linewidth=3,linestyle='dashed')
plt.plot(cfg.index_insert-1+5/16,entropies_orig[0], marker="o", markersize=16, markeredgecolor="tab:blue", markerfacecolor="tab:blue")
plt.plot(np.arange(0,len(cfg.outputdims)),entropies, linestyle='solid', linewidth=3, label='Embedded data')
plt.xticks(np.insert(np.arange(0,int(len(cfg.outputdims))).astype('float'),cfg.index_insert,cfg.index_insert-1+5/16),xticks,fontsize=cfg.TITLE_SIZE-4)
plt.yticks(fontsize=cfg.TITLE_SIZE-2)
plt.xlabel('Dimension of the embedding',fontsize=cfg.TITLE_SIZE)
plt.ylabel('Entropy',fontsize=cfg.TITLE_SIZE)
plt.title('Entropy in behavior assignment',fontsize=cfg.TITLE_SIZE)
plt.grid(True)
plt.legend(fontsize=cfg.TITLE_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(cfg.result_dir,'evaluation','entropies.png'),dpi=300)
plt.show()

# BIC
plt.plot(np.arange(0,len(cfg.outputdims)),BIC_orig, label='Original data',linewidth=3,linestyle='dashed')
plt.plot(cfg.index_insert-1+5/16,BIC_orig[0], marker="o", markersize=16, markeredgecolor="tab:blue", markerfacecolor="tab:blue")
plt.plot(np.arange(0,len(cfg.outputdims)),BIC, linestyle='solid', linewidth=3, label='Embedded data')
plt.xticks( np.insert(np.arange(0,int(len(cfg.outputdims))).astype('float'),cfg.index_insert,cfg.index_insert-1+5/16),xticks,fontsize=cfg.TITLE_SIZE-4)
plt.yticks(fontsize=cfg.TITLE_SIZE-7)
plt.xlabel('Dimension of the embedding',fontsize=cfg.TITLE_SIZE)
plt.ylabel('BIC',fontsize=cfg.TITLE_SIZE)
plt.title('BIC',fontsize=cfg.TITLE_SIZE)
plt.grid(True)
plt.legend(fontsize=cfg.TITLE_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(cfg.result_dir,'evaluation','BIC.png'),dpi=300)
plt.show()

