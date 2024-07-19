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
from sklearn.mixture import GaussianMixture
import pandas
import matplotlib.pyplot as plt
from config import cfg
import umap
import scipy.io as sio
from scipy import stats
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances

sns.set(rc={'figure.figsize':(11.7,8.27)})

# 1. GMM en el espacio original para la detección del número de comportamientos
# Read and organize the data
#NxTxF
data = sio.loadmat(os.path.join(cfg.data_dir,'cnic_dataset.mat'))['features_lstm']
ids=data[:,:,cfg.n_features:]
data=data[:,:,:cfg.n_features]
train_data=data[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.train_groups)),int((data.shape[1]-1)/2),:].copy().squeeze()
test_data=data[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.test_groups)),int((data.shape[1]-1)/2),:].copy().squeeze()
all_data=data[:,int((data.shape[1]-1)/2),:]
ids_train=ids[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.train_groups)),int((data.shape[1]-1)/2),:].copy()
ids_test=ids[np.in1d(ids[:,int(data.shape[1]/2),0],np.array(cfg.test_groups)),int((data.shape[1]-1)/2),:].copy()
group=ids[:,int((data.shape[1]-1)/2),0]
n_groups=int(group.max())

# Normalize the samples
mean=sio.loadmat(os.path.join(cfg.data_dir,'mean_features.mat'))['mean_lstm']
std=sio.loadmat(os.path.join(cfg.data_dir,'std_features.mat'))['std_lstm']

train_data_norm=(train_data-np.repeat(mean,train_data.shape[0],axis=0))/np.repeat(std,train_data.shape[0],axis=0)
test_data_norm=(test_data-np.repeat(mean,test_data.shape[0],axis=0))/np.repeat(std,test_data.shape[0],axis=0)
all_data_norm=(all_data-np.repeat(mean,all_data.shape[0],axis=0))/np.repeat(std,all_data.shape[0],axis=0)

## Correlation
if not os.path.exists(os.path.join(cfg.result_dir, 'analysis', 'original_data')):
    os.makedirs(os.path.join(cfg.result_dir,'analysis', 'original_data'))
linear_relationships=np.zeros((cfg.n_features,cfg.n_features),dtype='float')
for i in range(cfg.n_features):
    for j in range(cfg.n_features):
        pred=LinearRegression().fit(all_data_norm[:,i].reshape(-1, 1),all_data_norm[:,j]).predict(all_data_norm[:,i].reshape(-1, 1))
        linear_relationships[i,j]=r2_score(all_data_norm[:,j],pred)
        
sns.set(font_scale=0.7)
linear_relationships = pd.DataFrame(linear_relationships,columns=cfg.categories_orig[:-1],index=cfg.categories_orig[:-1]).round(2)
sns.heatmap(linear_relationships, annot=True, vmax=1, vmin=0, cmap='Blues')
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'linear_relationships_map.png'),bbox_inches='tight', dpi=300)
plt.show()
sns.set(font_scale=1.0)


sns.set(font_scale=0.7)
corr_matrix = pd.DataFrame(all_data,columns=cfg.categories_orig[:-1]).corr().round(2)
sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'corr_map.png'),bbox_inches='tight', dpi=300)
plt.show()
sns.set(font_scale=1.0)

palette = sns.color_palette("bright", int(cfg.K_selected))
palette_group = sns.color_palette("colorblind", int(group.max()))


cluster_agg=GaussianMixture(n_components=cfg.K_selected,n_init=10,random_state=cfg.SEED)
cluster_agg.fit(train_data_norm)
cidx_orig_train=cluster_agg.predict(train_data_norm)
cidx_orig_test=cluster_agg.predict(test_data_norm)
cidx_orig=cluster_agg.predict(all_data_norm)
probs_orig=cluster_agg.predict_proba(all_data_norm)
    
if (not os.path.exists(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'DB_score_behavior_' + cfg.alg + '.npy'))):
    # Analyze the Davies-Bouldin score for all the parameter combinations
    DB_score = np.zeros((cfg.param1.shape[0], cfg.param2.shape[0]), dtype='float')
    for i in range(len(cfg.param1)):
        for j in range(len(cfg.param2)):
            print('---' + str(i) + '---' + str(j))
            if (cfg.alg == 'tsne'):
                X_embedded = TSNE(n_components=2, perplexity=cfg.param1[i], early_exaggeration=cfg.param2[j],
                                  learning_rate=200.0, n_iter=1000,
                                  n_iter_without_progress=300, metric='euclidean', random_state=cfg.SEED).fit_transform(all_data)
            elif (cfg.alg == 'umap'):
                X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[i]),
                                        min_dist=int(cfg.param2[j]),
                                        metric='euclidean', random_state=cfg.SEED).fit_transform(all_data)
    
            # Lower value, better clustering
            DB_score[i, j] = davies_bouldin_score(X_embedded, cidx_orig.ravel())
    
    np.save(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'DB_score_behavior_' + cfg.alg + '.npy'), DB_score)
else:
    DB_score = np.load(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'DB_score_behavior_' + cfg.alg + '.npy'))

# Graphical representation of the behavior proportion in the groups
histograms = np.zeros((n_groups, int(cidx_orig.max()+1)), dtype='float')
for i in range(1, int(group.max() + 1)):
    cluster = cidx_orig[np.where(group == i)]
    for j in range(0,int(cidx_orig.max()+1)):
        histograms[int(i - 1), j] = np.sum(cluster == j) / cluster.shape[0]
ind = np.arange(start=1, stop=int(group.max() + 1)) * 0.8
width = 0.5
p = []
leg = []
for i in range(cidx_orig.max()+1):
    leg.append('behavior ' + str(i + 1))
    if (i == 0):
        bottom = np.zeros((n_groups,), dtype='float')
    else:
        bottom = np.zeros((n_groups,), dtype='float')
        for j in range(i):
            bottom += histograms[:, j]
    p.append(plt.bar(ind, histograms[:, i], width, bottom=bottom, color=palette[i], edgecolor="gray"))
plt.ylabel('Behavior proportions in groups', fontsize=16)
plt.xticks(ind, cfg.group_ids, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(tuple(p), tuple(leg), prop={'size': 14},loc="center right")
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'Stacked_behavior_proportion.png'), dpi=300)
plt.clf()

width = 1.0 / (cidx_orig.max()+1 + 2)
rr = []
for i in range(cidx_orig.max()+1):
    if (i == 0):
        rr.append(np.arange(len(histograms[:, 0])) + width / 2 + 0.1 * width)
    else:
        aux = [x + width + 0.1 * width for x in rr[-1]]
        rr.append(aux)

# Make the plot
p = []
for i in range(cidx_orig.max()+1):
    p.append(plt.bar(rr[i], histograms[:, i], color=palette[i], width=width, edgecolor='gray', label=leg[i]))

# Add xticks on the middle of the group bars
plt.ylabel('Behavior proportions in groups', fontsize=12)
plt.legend(tuple(p), tuple(leg), prop={'size': 12})
plt.xticks([i + 0.22 + n_groups / 2 * 0.11 for i in range(n_groups)], cfg.group_ids, fontsize=12)
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'Independent_behavior_proportion.png'), dpi=300)
plt.clf()

# For the best case of Davies-Bouldin, obtain the t-SNE or UMAP graphs
ind = np.unravel_index(np.argmin(DB_score, axis=None), DB_score.shape)
if (cfg.alg == 'tsne'):
    # t-SNE
    X_embedded = TSNE(n_components=2, perplexity=cfg.param1[ind[0]],
                      early_exaggeration=cfg.param2[ind[1]], learning_rate=200.0, n_iter=1000,
                      n_iter_without_progress=300, metric='euclidean', random_state=cfg.SEED).fit_transform(all_data)
elif (cfg.alg == 'umap'):
    # UMAP
    X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[ind[0]]), min_dist=int(cfg.param2[ind[1]]),
                           metric='euclidean', random_state=cfg.SEED).fit_transform(all_data)

sns_plot = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cidx_orig.ravel(), s=cfg.POINT_SIZE,
                           legend='full', alpha=cfg.ALPHA, palette=palette)
new_labels = []
for i in range(0, int(cidx_orig.max() + 1)):
    # replace labels
    new_labels.append('behavior ' + str(int(i+1)))
for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
sns_plot.figure.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', cfg.alg + '_behaviors.png'))
plt.clf()

sns_plot = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=group.ravel(), s=cfg.POINT_SIZE,
                           legend='full', alpha=cfg.ALPHA, palette=palette_group)
new_labels = []
for i in range(0, int(group.max())):
    # replace labels
    new_labels.append(cfg.group_ids[i])
for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
sns_plot.figure.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', cfg.alg + '_groups.png'))
plt.clf()

## Group comparison

data_spyder=np.zeros((n_groups,all_data.shape[1]),dtype='float')
error_spyder=np.zeros((n_groups,all_data.shape[1]),dtype='float')

for i in range(all_data.shape[1]):
    feat=all_data[:,i]
    iqr=stats.iqr(feat)
    ids=np.where(np.logical_and(feat>np.percentile(feat,25)-1.5*iqr,feat<np.percentile(feat,75)+1.5*iqr))[0]
    feat=feat[ids]
    feat=feat-min(feat)
    feat=feat/max(feat)
    for j in range(1,n_groups+1):
        group_feat=group.copy()
        group_feat=group_feat[ids]
        feat_group=feat[np.in1d(group_feat,np.array([j]))]
        data_spyder[int(j-1),i]=np.mean(feat_group)
        error_spyder[int(j-1),i]=np.std(feat_group)
        
data_spyder=np.concatenate((data_spyder,data_spyder[:,0][:,np.newaxis]),axis=1)
error_spyder=np.concatenate((error_spyder,error_spyder[:,0][:,np.newaxis]),axis=1)
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cfg.categories_orig))

plt.figure(figsize=(10, 10))
ax=plt.subplot(polar=True)
line1=plt.plot(label_loc, data_spyder[0,:], label=cfg.group_ids[0], color=palette_group[0])
line2=plt.plot(label_loc, data_spyder[1,:], label=cfg.group_ids[1], color=palette_group[1])
line3=plt.plot(label_loc, data_spyder[2,:], label=cfg.group_ids[2], color=palette_group[2])
line4=plt.plot(label_loc, data_spyder[3,:], label=cfg.group_ids[3], color=palette_group[3])
plt.title('Group comparison', size=20)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=cfg.categories_orig)
plt.legend(loc=(-0.1,1.0))
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'group_comparison.png'), dpi=300)  
plt.show()

data_orig_min=np.min(all_data,axis=0)[np.newaxis,:]
data_orig=all_data-np.repeat(data_orig_min,all_data.shape[0],axis=0)
data_orig_max=np.max(data_orig,axis=0)[np.newaxis,:]
data_orig=data_orig/np.repeat(data_orig_max,data_orig.shape[0],axis=0)
data_orig=np.concatenate((data_orig,data_orig[:,0][:,np.newaxis]),axis=1)

data1=np.mean(data_orig[np.in1d(cidx_orig,np.array([0])),:],axis=0)
data2=np.mean(data_orig[np.in1d(cidx_orig,np.array([1])),:],axis=0)
data3=np.mean(data_orig[np.in1d(cidx_orig,np.array([2])),:],axis=0)
data4=np.mean(data_orig[np.in1d(cidx_orig,np.array([3])),:],axis=0)
data5=np.mean(data_orig[np.in1d(cidx_orig,np.array([4])),:],axis=0)
data6=np.mean(data_orig[np.in1d(cidx_orig,np.array([5])),:],axis=0)

plt.figure(figsize=(8, 8))
ax=plt.subplot(polar=True)
line1=plt.plot(label_loc, data1, label='behavior 1')
line2=plt.plot(label_loc, data2, label='behavior 2')
line3=plt.plot(label_loc, data3, label='behavior 3')
line4=plt.plot(label_loc, data4, label='behavior 4')
line5=plt.plot(label_loc, data5, label='behavior 5')
line6=plt.plot(label_loc, data6, label='behavior 6')
plt.title('Behavior comparison', size=20)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=cfg.categories_orig)
plt.legend(loc=(-0.1,0.9))
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'behavior_comparison.png'), dpi=300)
plt.show()

# Analize the behavior transitions and normalized displacements w.r.t. the closest cluster
neutrophils_by_trajectory = sio.loadmat(os.path.join(cfg.data_dir, 'neutrophils_by_trajectory.mat'))['neutrophils_by_trajectory']
neutrophils_by_timestamp = sio.loadmat(os.path.join(cfg.data_dir, 'neutrophils_by_timestamp.mat'))['neutrophils_by_timestamp']
id_neut=np.unique(neutrophils_by_trajectory)
id_sort=np.argsort(neutrophils_by_trajectory,axis=0).ravel()
neutrophils_by_trajectoryS=neutrophils_by_trajectory[id_sort]
neutrophils_by_timestampS=neutrophils_by_timestamp[id_sort]
X_embeddedS=X_embedded[id_sort,:]
X_fullS=all_data_norm[id_sort,:]
cidx_origS=cidx_orig[id_sort].copy()
probs_origS=probs_orig[id_sort,:].copy()
groupS=group[id_sort].copy()
pos_emb=[]
disp_emb_orig=[]
disp_timestamps_orig=[]
behaviors_per_traj=[]
behaviors_per_traj_orig1=[]
behaviors_per_traj_orig2=[]
id_cell1_orig=[]
id_cell2_orig=[]
id_traj_orig=[]
id_group_orig=[]
behavior_prob_orig=[]
valids_orig=0
duration_orig=[]

for i in range(id_neut.shape[0]):
    id_traj=np.where(neutrophils_by_trajectoryS==id_neut[i])[0]
    timestamps_traj=neutrophils_by_timestampS[np.where(neutrophils_by_trajectoryS==id_neut[i])[0]]
    id_sort=np.argsort(timestamps_traj,axis=0).ravel()
    timestamps_traj=timestamps_traj[id_sort]
    X_embedded_traj=X_embeddedS[id_traj,:].copy()[id_sort,:]
    id_group_orig.append(groupS[id_traj][:-1])
    X_full_traj=X_fullS[id_traj,:].copy()[id_sort,:]
    pos_emb.append(X_full_traj)
    behaviors_per_traj.append(cidx_origS[id_traj].copy()[id_sort]+1)
    aux=cidx_origS[id_traj].copy()[id_sort]+1
    dur=1
    for j in range(len(aux)-1):
        if (not aux[j]==aux[j+1]):
            id_aux=np.where(np.logical_and(timestamps_traj>timestamps_traj[j],timestamps_traj<=timestamps_traj[j]+cfg.PTH))[0]
            if (aux[id_aux].size>0):
                if (np.unique(aux[id_aux]).size==1):
                    valids_orig=valids_orig+1
            else:
                valids_orig=valids_orig+1
            duration_orig.append(dur)
            dur=1
        else:
            dur=dur+1
    behavior_prob_orig.append(probs_origS[id_traj,:].copy()[id_sort,:])
    disp_emb_orig.append(np.sqrt(np.sum(np.diff(pos_emb[-1],axis=0)**2,axis=1)))
    behaviors_per_traj_orig1.append(cidx_origS[id_traj].copy()[id_sort][:-1]+1)
    behaviors_per_traj_orig2.append(cidx_origS[id_traj].copy()[id_sort][1:]+1)
    disp_timestamps_orig.append(np.diff(timestamps_traj,axis=0))
    id_cell1_orig.append(timestamps_traj[:-1])
    id_cell2_orig.append(timestamps_traj[1:])
    id_traj_orig.append(id_neut[i]*np.ones_like(timestamps_traj)[:-1])

# Get displacements for each cell
disp_emb_orig=np.concatenate(disp_emb_orig,axis=0)
behaviors_per_traj_orig1=np.concatenate(behaviors_per_traj_orig1,axis=0)
behaviors_per_traj_orig2=np.concatenate(behaviors_per_traj_orig2,axis=0)
disp_timestamps_orig=np.concatenate(disp_timestamps_orig,axis=0)
id_traj_orig=np.concatenate(id_traj_orig,axis=0)
id_cell1_orig=np.concatenate(id_cell1_orig,axis=0)
id_cell2_orig=np.concatenate(id_cell2_orig,axis=0)
behaviors_per_traj=np.concatenate(behaviors_per_traj,axis=0)
behavior_prob_orig=np.concatenate(behavior_prob_orig,axis=0)
id_group_orig=np.concatenate(id_group_orig,axis=0)
duration_orig=np.array(duration_orig)

behavior_t1_orig=[]
transition_t1_orig=[]
transition_t2_orig=[]
behavior_t2_orig=[]
for i in range(behaviors_per_traj.shape[0]-1):
    if (neutrophils_by_trajectoryS[i]==neutrophils_by_trajectoryS[i+1]):
        if (not behaviors_per_traj[i]==behaviors_per_traj[i+1]):
            behavior_t1_orig.append(behavior_prob_orig[i,behaviors_per_traj[i]-1])
            behavior_t2_orig.append(behavior_prob_orig[i+1,behaviors_per_traj[i]-1])
            transition_t1_orig.append(behavior_prob_orig[i,behaviors_per_traj[i+1]-1])
            transition_t2_orig.append(behavior_prob_orig[i+1,behaviors_per_traj[i+1]-1])

med_disp_emb_orig=disp_emb_orig/(disp_timestamps_orig.ravel())

n_cells_not_changing_behavior_orig=np.sum(behaviors_per_traj_orig1==behaviors_per_traj_orig2)
n_cells_changing_behavior_orig=np.sum(behaviors_per_traj_orig1!=behaviors_per_traj_orig2)
n_cells_not_changing_behavior_orig_group=np.zeros((n_groups,),dtype='int')
n_cells_changing_behavior_orig_group=np.zeros((n_groups,),dtype='int')
for i in range(n_groups):
    n_cells_not_changing_behavior_orig_group[i]=np.sum(behaviors_per_traj_orig1[np.where(id_group_orig==int(i+1))[0]]==behaviors_per_traj_orig2[np.where(id_group_orig==int(i+1))[0]])
    n_cells_changing_behavior_orig_group[i]=np.sum(behaviors_per_traj_orig1[np.where(id_group_orig==int(i+1))[0]]!=behaviors_per_traj_orig2[np.where(id_group_orig==int(i+1))[0]])    


centroids=np.zeros((cfg.K_selected,all_data_norm.shape[1]),dtype='float')
for i in range(cfg.K_selected):
    centroids[i,:]=np.mean(all_data_norm[np.where(cidx_orig==i)[0],:],axis=0)
cluster_distance_emb = euclidean_distances(centroids)
cluster_distance_emb = cluster_distance_emb+1e6*np.eye(cluster_distance_emb.shape[0])
cluster_distance_emb = np.min(cluster_distance_emb,axis=1)

for i in range(med_disp_emb_orig.shape[0]):
    med_disp_emb_orig[i]=med_disp_emb_orig[i]/cluster_distance_emb[behaviors_per_traj_orig1[i]-1]

H=np.histogram(med_disp_emb_orig[behaviors_per_traj_orig1==behaviors_per_traj_orig2], bins=10,range=(0,np.max(med_disp_emb_orig)))
plt.bar((H[1][:-1]+H[1][1:])/2, H[0], width=0.9*(H[1][1]-H[1][0]), edgecolor="gray")
H=np.histogram(med_disp_emb_orig[behaviors_per_traj_orig1!=behaviors_per_traj_orig2], bins=10,range=(0,np.max(med_disp_emb_orig)))
plt.bar((H[1][:-1]+H[1][1:])/2, H[0], width=0.45*(H[1][1]-H[1][0]), edgecolor="red")
plt.ylabel('Number of cells', fontsize=12)
plt.xlabel('Relative displacement respect to the closest cluster', fontsize=12)
plt.legend(['not changing behavior ('+str(n_cells_not_changing_behavior_orig)+')','changing behavior ('+str(n_cells_changing_behavior_orig)+')'])
# plt.yscale("log")
plt.savefig(os.path.join(cfg.result_dir, 'analysis', 'original_data', 'behavior_transitions_and_displacements.png'), dpi=300)
plt.clf()



# # Embedding data
embs=[16]
for id_emb, N_emb in enumerate(embs):
    folder=os.path.join('analysis', 'embedding_data_'+str(N_emb))
    if not os.path.exists(os.path.join(cfg.result_dir, folder)):
        os.mkdir(os.path.join(cfg.result_dir,folder))
    
    model_path=cfg.arch+'_g_{}_gtraj_{}_lr_{}_bs_{}_pnum_{}_traj_{}_out_{}'.format(cfg.beta, cfg.beta_traj, cfg.lr, cfg.batch_size,cfg.pnum, cfg.pnum_traj, str(N_emb))
    data = sio.loadmat(os.path.join(cfg.train_dir,model_path,'embedding_best.mat'))['embedding'].T
    group=data[:,-1].astype('int')
    data=data[:,:-1]
    n_groups=int(group.max())
    group_ids=['Group '+str(int(i+1)) for i in range(n_groups)]
    
    train_data=data[np.in1d(group,np.array(cfg.train_groups)),:].copy()
    test_data=data[np.in1d(group,np.array(cfg.test_groups)),:].copy()
    # Normalize the samples
    mean=np.mean(data,axis=0)[None,:]
    std=np.std(data,axis=0)[None,:]
    train_data_norm=(train_data-np.repeat(mean,train_data.shape[0],axis=0))/np.repeat(std,train_data.shape[0],axis=0)
    test_data_norm=(test_data-np.repeat(mean,test_data.shape[0],axis=0))/np.repeat(std,test_data.shape[0],axis=0)
    all_data_norm=(data-np.repeat(mean,data.shape[0],axis=0))/np.repeat(std,data.shape[0],axis=0)

    # Correlation matrix
    categories = ['features '+str((int(i+1))) for i in range(data.shape[1])]
    categories.append('')
    if (N_emb>16):
        sns.set(rc={'figure.figsize':(11.7*3,8.27*3)})
        
    linear_relationships=np.zeros((data.shape[1],data.shape[1]),dtype='float')
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            pred=LinearRegression().fit(all_data_norm[:,i].reshape(-1, 1),all_data_norm[:,j]).predict(all_data_norm[:,i].reshape(-1, 1))
            linear_relationships[i,j]=r2_score(all_data_norm[:,j],pred)
            
    sns.set(font_scale=0.7)
    linear_relationships = pd.DataFrame(linear_relationships,columns=categories[:-1],index=categories[:-1]).round(2)
    sns.heatmap(linear_relationships, annot=True, vmax=1, vmin=0, cmap='Blues')
    plt.savefig(os.path.join(cfg.result_dir, folder, 'linear_relationships_map.png'),bbox_inches='tight', dpi=300)
    plt.show()
    
    corr_matrix = pd.DataFrame(data,columns=categories[:-1]).corr().round(2)
    sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
    plt.savefig(os.path.join(cfg.result_dir, folder,'corr_map_emb.png'), bbox_inches='tight', dpi=300)
    plt.show()
    sns.set(font_scale=1.0)
    if (N_emb>16):
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        
    if (cfg.REMAP):
        # We load the clustering and probabilities for our model
        cidx_embed=sio.loadmat(os.path.join(cfg.result_dir, folder,'behaviors_emb.mat'))['behavior'].ravel()-1
        probs_embed=sio.loadmat(os.path.join(cfg.result_dir, folder,'probs_embed.mat'))['probs_embed']

        # We built a GMM with the assignations from the paper, just in case the implementation in newer versions
        # of Python does not provide the same clustering
        # Extract the train samples
        cidx_embed_train=cidx_embed[np.in1d(group,np.array(cfg.train_groups))]
        mus=np.zeros((cfg.K_selected,all_data_norm.shape[1]),dtype='float')
        precs=np.zeros((cfg.K_selected,all_data_norm.shape[1],all_data_norm.shape[1]),dtype='float')
        pis=np.zeros((cfg.K_selected,),dtype='float')
        for k in range(cfg.K_selected):
            mus[k,:]=np.mean(train_data_norm[cidx_embed_train==k,:],axis=0)
            precs[k,...]=np.linalg.inv(np.cov(train_data_norm[cidx_embed_train==k,:].T))
            pis[k]=np.sum(cidx_embed_train==k)/cidx_embed_train.size
            
        cluster_agg=GaussianMixture(n_components=cfg.K_selected,n_init=10,random_state=cfg.SEED,means_init=mus, precisions_init=precs, weights_init=pis)
        cluster_agg.fit(train_data_norm)
        cidx_embed_train=cluster_agg.predict(train_data_norm)
        cidx_embed_test=cluster_agg.predict(test_data_norm)
        new_cidx_embed=cluster_agg.predict(all_data_norm)
        new_probs_embed=cluster_agg.predict_proba(all_data_norm)
        print('Coincidence between the new behaviors and behaviors from the paper')
        print(np.sum(new_cidx_embed==cidx_embed)/np.size(cidx_embed))
        print('Mean absolute error in the probabilities')
        print(np.mean(new_probs_embed==probs_embed)/np.size(probs_embed))
    else:
        cluster_agg=GaussianMixture(n_components=cfg.K_selected,n_init=10,random_state=cfg.SEED)
        cluster_agg.fit(train_data_norm)
        cidx_embed_train=cluster_agg.predict(train_data_norm)
        cidx_embed_test=cluster_agg.predict(test_data_norm)
        cidx_embed=cluster_agg.predict(all_data_norm)
        cidx_embed2=cidx_embed+1
        sio.savemat(os.path.join(cfg.result_dir, folder,'behaviors_emb.mat'),{'behavior':cidx_embed2})
    
    # Analyze the Davies-Bouldin score for all the parameter combinations
    if (not os.path.exists(os.path.join(cfg.result_dir, folder, 'DB_score_behavior_' + cfg.alg + '_emb.npy'))):
        DB_score = np.zeros((cfg.param1.shape[0], cfg.param2.shape[0]), dtype='float')
        for i in range(len(cfg.param1)):
            for j in range(len(cfg.param2)):
                print('---' + str(i) + '---' + str(j))
                if (cfg.alg == 'tsne'):
                    X_embedded = TSNE(n_components=2, perplexity=cfg.param1[i], early_exaggeration=cfg.param2[j],
                                      learning_rate=200.0, n_iter=1000,
                                      n_iter_without_progress=300, metric='euclidean', random_state=cfg.SEED).fit_transform(data)
                elif (cfg.alg == 'umap'):
                    X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[i]),
                                            min_dist=int(cfg.param2[j]),
                                            metric='euclidean', random_state=cfg.SEED).fit_transform(data)
        
                # Lower value, better clustering
                DB_score[i, j] = davies_bouldin_score(X_embedded, cidx_embed.ravel())
        
        np.save(os.path.join(cfg.result_dir, folder, 'DB_score_behavior_' + cfg.alg + '_emb.npy'), DB_score)
    else:
        DB_score = np.load(os.path.join(cfg.result_dir, folder, 'DB_score_behavior_' + cfg.alg + '_emb.npy'))
    
    # For the best case of Davies-Bouldin values, obtain the t-SNE or UMAP graphs
    # Davies Bouldin
    ind = np.unravel_index(np.argmin(DB_score, axis=None), DB_score.shape)
    DB_score[ind[0], ind[1]] = np.inf  # Delete for the next iteration
    if (cfg.alg == 'tsne'):
        # t-SNE
        X_embedded = TSNE(n_components=2, perplexity=cfg.param1[ind[0]],
                          early_exaggeration=cfg.param2[ind[1]], learning_rate=200.0, n_iter=1000,
                          n_iter_without_progress=300, metric='euclidean', random_state=cfg.SEED).fit_transform(data)
    elif (cfg.alg == 'umap'):
        # UMAP
        X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[ind[0]]), min_dist=int(cfg.param2[ind[1]]),
                               metric='euclidean', random_state=cfg.SEED).fit_transform(data)

    if (cfg.REMAP):
        sns_plot = sns.scatterplot(y=X_embedded[:, 0], x=-X_embedded[:, 1], hue=cidx_embed.ravel(), s=cfg.POINT_SIZE,
                                   legend='full', alpha=cfg.ALPHA, palette=palette)
        new_labels = []
        for i in range(0, int(cidx_embed.max() + 1)):
            # replace labels
            new_labels.append('behavior ' + str(int(i+1)))
        for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
        sns_plot.figure.savefig(os.path.join(cfg.result_dir, folder, cfg.alg + '_behaviors_emb.png'))
        plt.clf()
        
        sns_plot = sns.scatterplot(y=X_embedded[:, 0], x=-X_embedded[:, 1], hue=group.ravel(), s=cfg.POINT_SIZE,
                                   legend='full', alpha=cfg.ALPHA, palette=palette_group)
        new_labels = []
        for i in range(0, int(group.max())):
            # replace labels
            new_labels.append(cfg.group_ids[i])
        for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
        sns_plot.figure.savefig(os.path.join(cfg.result_dir, folder, cfg.alg + '_groups_emb.png'))
        plt.clf()
    else:
        if (cfg.alg == 'tsne'):
            # t-SNE
            X_embedded = TSNE(n_components=2, perplexity=cfg.param1[ind[0]],
                              early_exaggeration=cfg.param2[ind[1]], learning_rate=200.0, n_iter=1000,
                              n_iter_without_progress=300, metric='euclidean', random_state=cfg.SEED).fit_transform(data)
        elif (cfg.alg == 'umap'):
            # UMAP
            X_embedded = umap.UMAP(n_neighbors=int(cfg.param1[ind[0]]), min_dist=int(cfg.param2[ind[1]]),
                                   metric='euclidean', random_state=cfg.SEED).fit_transform(data)

        sns_plot = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=cidx_embed.ravel(), s=cfg.POINT_SIZE,
                                   legend='full', alpha=cfg.ALPHA, palette=palette)
        new_labels = []
        for i in range(0, int(cidx_embed.max() + 1)):
            # replace labels
            new_labels.append('behavior ' + str(int(i+1)))
        for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
        sns_plot.figure.savefig(os.path.join(cfg.result_dir, folder, cfg.alg + '_behaviors_emb.png'))
        plt.clf()
        
        sns_plot = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=group.ravel(), s=cfg.POINT_SIZE,
                                   legend='full', alpha=cfg.ALPHA, palette=palette_group)
        new_labels = []
        for i in range(0, int(group.max())):
            # replace labels
            new_labels.append(cfg.group_ids[i])
        for t, l in zip(sns_plot.get_legend().texts, new_labels):  t.set_text(l)
        sns_plot.figure.savefig(os.path.join(cfg.result_dir, folder, cfg.alg + '_groups_emb.png'))
        plt.clf()
    
    # Graphical representation of the behavior proportion in the groups
    histograms = np.zeros((n_groups, int(cidx_embed.max()+1)), dtype='float')
    for i in range(1, int(group.max() + 1)):
        cluster = cidx_embed[np.where(group == i)]
        for j in range(0,int(cidx_embed.max()+1)):
            histograms[int(i - 1), j] = np.sum(cluster == j) / cluster.shape[0]
    ind = np.arange(start=1, stop=int(group.max() + 1)) * 0.8
    width = 0.5
    p = []
    leg = []
    for i in range(cidx_embed.max()+1):
        leg.append('behavior ' + str(i + 1))
        if (i == 0):
            bottom = np.zeros((n_groups,), dtype='float')
        else:
            bottom = np.zeros((n_groups,), dtype='float')
            for j in range(i):
                bottom += histograms[:, j]
        p.append(plt.bar(ind, histograms[:, i], width, bottom=bottom, color=palette[i], edgecolor="gray"))
    plt.ylabel('Behavior proportion in groups', fontsize=16)
    plt.xticks(ind, cfg.group_ids, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(tuple(p), tuple(leg), prop={'size': 14},loc="center right")
    plt.savefig(os.path.join(cfg.result_dir, folder, 'Stacked_behavior_proportion_emb.png'), dpi=300)
    plt.clf()
    
    width = 1.0 / (cidx_embed.max()+1 + 2)
    rr = []
    for i in range(cidx_embed.max()+1):
        if (i == 0):
            rr.append(np.arange(len(histograms[:, 0])) + width / 2 + 0.1 * width)
        else:
            aux = [x + width + 0.1 * width for x in rr[-1]]
            rr.append(aux)
    
    # Make the plot
    p = []
    for i in range(cidx_embed.max()+1):
        p.append(plt.bar(rr[i], histograms[:, i], color=palette[i], width=width, edgecolor='gray', label=leg[i]))
    
    # Add xticks on the middle of the group bars
    plt.ylabel('Behavior proportion in groups', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(tuple(p), tuple(leg), prop={'size': 16})
    plt.xticks([i + 0.22 + n_groups / 2 * 0.11 for i in range(n_groups)], cfg.group_ids, fontsize=16)
    plt.savefig(os.path.join(cfg.result_dir, folder, 'Independent_behavior_proportion_emb.png'), dpi=300)
    plt.clf()
    
    
    ## Group comparison

    data_spyder=np.zeros((n_groups,data.shape[1]),dtype='float')
    error_spyder=np.zeros((n_groups,data.shape[1]),dtype='float')
    
    for i in range(data.shape[1]):
        feat=data[:,i]
        iqr=stats.iqr(feat)
        ids=np.where(np.logical_and(feat>np.percentile(feat,25)-1.5*iqr,feat<np.percentile(feat,75)+1.5*iqr))[0]
        feat=feat[ids]
        feat=feat-min(feat)
        feat=feat/max(feat)
        for j in range(1,n_groups+1):
            group_feat=group.copy()
            group_feat=group_feat[ids]
            feat_group=feat[np.in1d(group_feat,np.array([j]))]
            data_spyder[int(j-1),i]=np.mean(feat_group)
            error_spyder[int(j-1),i]=np.std(feat_group)
            
    data_spyder=np.concatenate((data_spyder,data_spyder[:,0][:,np.newaxis]),axis=1)
    error_spyder=np.concatenate((error_spyder,error_spyder[:,0][:,np.newaxis]),axis=1)
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(8, 8))
    ax=plt.subplot(polar=True)
    line1=plt.plot(label_loc, data_spyder[0,:], label=cfg.group_ids[0], color=palette_group[0])
    line2=plt.plot(label_loc, data_spyder[1,:], label=cfg.group_ids[1], color=palette_group[1])
    line3=plt.plot(label_loc, data_spyder[2,:], label=cfg.group_ids[2], color=palette_group[2])
    line4=plt.plot(label_loc, data_spyder[3,:], label=cfg.group_ids[3], color=palette_group[3])
    plt.title('Group comparison', size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend(loc=(-0.1,1.0))
    plt.savefig(os.path.join(cfg.result_dir, folder, 'group_comparison_emb.png'), dpi=300)  
    plt.show()
    
    data_orig_min=np.min(data,axis=0)[np.newaxis,:]
    data_orig=data-np.repeat(data_orig_min,data.shape[0],axis=0)
    data_orig_max=np.max(data_orig,axis=0)[np.newaxis,:]
    data_orig=data_orig/np.repeat(data_orig_max,data_orig.shape[0],axis=0)
    data_orig=np.concatenate((data_orig,data_orig[:,0][:,np.newaxis]),axis=1)
    
    data1=np.mean(data_orig[np.in1d(cidx_embed,np.array([0])),:],axis=0)
    data2=np.mean(data_orig[np.in1d(cidx_embed,np.array([1])),:],axis=0)
    data3=np.mean(data_orig[np.in1d(cidx_embed,np.array([2])),:],axis=0)
    data4=np.mean(data_orig[np.in1d(cidx_embed,np.array([3])),:],axis=0)
    data5=np.mean(data_orig[np.in1d(cidx_embed,np.array([4])),:],axis=0)
    data6=np.mean(data_orig[np.in1d(cidx_embed,np.array([5])),:],axis=0)
    
    plt.figure(figsize=(10,10))
    ax=plt.subplot(polar=True)
    line1=plt.plot(label_loc, data1, label='behavior 1',color=palette[0])
    line2=plt.plot(label_loc, data2, label='behavior 2',color=palette[1])
    line3=plt.plot(label_loc, data3, label='behavior 3',color=palette[2])
    line4=plt.plot(label_loc, data4, label='behavior 4',color=palette[3])
    line5=plt.plot(label_loc, data5, label='behavior 5',color=palette[4])
    line6=plt.plot(label_loc, data6, label='behavior 6',color=palette[5])
    plt.title('Behavior comparison', size=20)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories,fontsize=14)
    plt.legend(loc=(-0.1,0.9),fontsize=12)
    plt.savefig(os.path.join(cfg.result_dir, folder, 'behavior_comparison_emb.png'), dpi=300)
    plt.show()
    
    ## Analyze the number of behavior transitions and normalized displacements w.r.t. the closest cluster
    neutrophils_by_trajectory = sio.loadmat(os.path.join(cfg.data_dir,'neutrophils_by_trajectory.mat'))['neutrophils_by_trajectory']
    neutrophils_by_timestamp = sio.loadmat(os.path.join(cfg.data_dir,'neutrophils_by_timestamp.mat'))['neutrophils_by_timestamp']
    id_neut=np.unique(neutrophils_by_trajectory)
    id_sort=np.argsort(neutrophils_by_trajectory,axis=0).ravel()
    neutrophils_by_trajectoryS=neutrophils_by_trajectory[id_sort]
    neutrophils_by_timestampS=neutrophils_by_timestamp[id_sort]
    X_embeddedS=X_embedded[id_sort,:]
    X_fullS=all_data_norm[id_sort,:]
    cidx_embedS=cidx_embed[id_sort].copy()
    probs_embedS=probs_embed[id_sort,:].copy()
    groupS=group[id_sort].copy()
    pos_emb=[]
    disp_emb=[]
    disp_timestamps=[]
    behaviors_per_traj=[]
    behaviors_per_traj1=[]
    behaviors_per_traj2=[]
    id_traj_emb=[]
    id_cell1=[]
    id_cell2=[]
    behavior_prob=[]
    id_group=[]
    valids=0
    valid_ids=[]
    duration=[]
    for i in range(id_neut.shape[0]):
        id_traj=np.where(neutrophils_by_trajectoryS==id_neut[i])[0]
        timestamps_traj=neutrophils_by_timestampS[np.where(neutrophils_by_trajectoryS==id_neut[i])[0]]
        id_sort=np.argsort(timestamps_traj,axis=0).ravel()
        timestamps_traj=timestamps_traj[id_sort]
        X_embedded_traj=X_embeddedS[id_traj,:].copy()[id_sort,:]
        X_full_traj=X_fullS[id_traj,:].copy()[id_sort,:]
        id_group.append(groupS[id_traj][:-1])
        pos_emb.append(X_full_traj)
        behaviors_per_traj.append(cidx_embedS[id_traj].copy()[id_sort]+1)
        aux=cidx_embedS[id_traj].copy()[id_sort]+1
        correct=np.ones_like(aux)
        dur=1
        for j in range(len(aux)-1):
            if (not aux[j]==aux[j+1]):
                id_aux=np.where(np.logical_and(timestamps_traj>timestamps_traj[j],timestamps_traj<=timestamps_traj[j]+cfg.PTH))[0]
                if (aux[id_aux].size>0):
                    if (np.unique(aux[id_aux]).size==1):
                        valids=valids+1
                    else:
                        correct[j]=0
                else:
                    valids=valids+1
                duration.append(dur)
                dur=1
            else:
                dur=dur+1
                    
        behavior_prob.append(probs_embedS[id_traj,:].copy()[id_sort,:])
        disp_emb.append(np.sqrt(np.sum(np.diff(pos_emb[-1],axis=0)**2,axis=1)))
        behaviors_per_traj1.append(cidx_embedS[id_traj].copy()[id_sort][:-1]+1)
        behaviors_per_traj2.append(cidx_embedS[id_traj].copy()[id_sort][1:]+1)
        disp_timestamps.append(np.diff(timestamps_traj,axis=0))
        id_cell1.append(timestamps_traj[:-1])
        id_cell2.append(timestamps_traj[1:])
        id_traj_emb.append(id_neut[i]*np.ones_like(timestamps_traj)[:-1])
        valid_ids.append(correct)
    
    # Get displacements for each cell
    disp_emb=np.concatenate(disp_emb,axis=0)
    behaviors_per_traj1=np.concatenate(behaviors_per_traj1,axis=0)
    behaviors_per_traj2=np.concatenate(behaviors_per_traj2,axis=0)
    disp_timestamps=np.concatenate(disp_timestamps,axis=0)
    id_cell1=np.concatenate(id_cell1,axis=0)
    id_cell2=np.concatenate(id_cell2,axis=0)
    id_traj_emb=np.concatenate(id_traj_emb,axis=0)
    behaviors_per_traj=np.concatenate(behaviors_per_traj,axis=0)
    behavior_prob=np.concatenate(behavior_prob,axis=0)
    id_group=np.concatenate(id_group,axis=0)
    valid_ids=np.concatenate(valid_ids,axis=0)
    duration=np.array(duration)
    behavior_t1=[]
    transition_t1=[]
    transition_t2=[]
    behavior_t2=[]
    for i in range(behaviors_per_traj.shape[0]-1):
        if (neutrophils_by_trajectoryS[i]==neutrophils_by_trajectoryS[i+1]):
            if (not behaviors_per_traj[i]==behaviors_per_traj[i+1]):
                behavior_t1.append(behavior_prob[i,behaviors_per_traj[i]-1])
                behavior_t2.append(behavior_prob[i+1,behaviors_per_traj[i]-1])
                transition_t1.append(behavior_prob[i,behaviors_per_traj[i+1]-1])
                transition_t2.append(behavior_prob[i+1,behaviors_per_traj[i+1]-1])
                
    med_disp_emb=disp_emb/(disp_timestamps.ravel())
    
    n_cells_not_changing_behavior=np.sum(behaviors_per_traj1==behaviors_per_traj2)
    n_cells_changing_behavior=np.sum(behaviors_per_traj1!=behaviors_per_traj2)
    n_cells_not_changing_behavior_group=np.zeros((n_groups,),dtype='int')
    n_cells_changing_behavior_group=np.zeros((n_groups,),dtype='int')
    for i in range(n_groups):
        n_cells_not_changing_behavior_group[i]=np.sum(behaviors_per_traj1[np.where(id_group==int(i+1))[0]]==behaviors_per_traj2[np.where(id_group==int(i+1))[0]])
        n_cells_changing_behavior_group[i]=np.sum(behaviors_per_traj1[np.where(id_group==int(i+1))[0]]!=behaviors_per_traj2[np.where(id_group==int(i+1))[0]])    

    cluster_distance_emb = euclidean_distances(cluster_agg.means_)
    cluster_distance_emb = cluster_distance_emb+1e6*np.eye(cluster_distance_emb.shape[0])
    cluster_distance_emb = np.min(cluster_distance_emb,axis=1)
        
    for i in range(med_disp_emb.shape[0]):
        med_disp_emb[i]=med_disp_emb[i]/cluster_distance_emb[behaviors_per_traj1[i]-1]
        
    H=np.histogram(med_disp_emb[behaviors_per_traj1==behaviors_per_traj2], bins=10,range=(0,np.max(med_disp_emb)))
    plt.bar((H[1][:-1]+H[1][1:])/2, H[0], width=0.9*(H[1][1]-H[1][0]), edgecolor="gray")
    H=np.histogram(med_disp_emb[behaviors_per_traj1!=behaviors_per_traj2], bins=10,range=(0,np.max(med_disp_emb)))
    plt.bar((H[1][:-1]+H[1][1:])/2, H[0], width=0.45*(H[1][1]-H[1][0]), edgecolor="red")
    plt.ylabel('Number of cells', fontsize=12)
    plt.xlabel('Relative displacement respect to the closest cluster', fontsize=12)
    plt.legend(['not changing behavior ('+str(n_cells_not_changing_behavior)+')','changing behavior ('+str(n_cells_changing_behavior)+')'])
    plt.savefig(os.path.join(cfg.result_dir, folder,  'behavior_transitions_and_displacements.png'), dpi=300)
    plt.clf()
    
    sns.set(font_scale=1.0)
    corr_matrix=np.zeros((int(len(cfg.categories_orig)-1),int(len(categories)-1)),dtype='float32')
    for i in range(len(cfg.categories_orig)-1):
        for j in range(len(categories)-1):
            corr_matrix[i,j]=np.corrcoef(all_data[:,i],data[:,j])[0,1]
            
    corr_matrix_sign=corr_matrix.copy()
    corr_matrix = pd.DataFrame(abs(corr_matrix),columns=categories[:-1],index=cfg.categories_orig[:-1]).round(3)
    sns.heatmap(corr_matrix, annot=corr_matrix_sign, vmax=1, vmin=0, center=0.5, cmap='Blues',annot_kws={"fontsize":9})
    plt.savefig(os.path.join(cfg.result_dir, folder, 'corr_map_embed_orig.png'),bbox_inches='tight', dpi=300)
    plt.show()
    sns.set(font_scale=1.0)


