#!/usr/bin/env python
# coding: utf-8

from numpy.random import randint
from numpy.random import rand


from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import Scripts.LevenClustering as levenClustering


# In[ ]:


### Genetic Algorithm functions

# tournament selection
def selection(pop, scores, ascending, k=2): ### k = how many chosen to be in the tournament
    # first random selection
    selection_ix = randint(len(pop))
    # second random selection
    ix = randint(len(pop))
    
    while selection_ix == ix:
        ix = randint(len(pop))
        
    ### Silhouette Score 
    if ascending :
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
            
    else:
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
               
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    p_random = rand(len(p1))
    for i in range(len(p_random)):
        # check for recombination
        # Uniform crossover
        if p_random[i] < r_cross:
            temp = c1[i]
            c1[i] = c2[i]
            c2[i] = temp
            
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

            
def fitness(df_x, n_clusters, s_score = 1):
    
    df_clustering = df_x.copy()
#     df_clustering['final_sequence'] = df_clustering['new_sequence']
    seq_list = df_clustering['new_sequence'].tolist()

    distance_matrix = levenClustering.dis_matrix_calc(seq_list)
    n_cluster = n_clusters
    linkage_matrix = levenClustering.hierarchical_clustering(distance_matrix)
    # select maximum number of clusters
    cluster_labels = fcluster(linkage_matrix, n_cluster, criterion='maxclust')
    # # hand-select an appropriate cut-off on the dendrogram
    # cluster_labels = fcluster(linkage_matrix, 240, criterion='distance')
    # print(np.unique(cluster_labels))
    df_clustering['Clusters'] = cluster_labels.tolist()
    
    fitness_score = 0
    
    if s_score == 1:
        fitness_score = silhouette_score(distance_matrix, cluster_labels)

    return df_clustering, fitness_score

