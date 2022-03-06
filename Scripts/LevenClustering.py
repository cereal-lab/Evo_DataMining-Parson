#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### hierarchical_clustering & calculating levenshtein distance

from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram
import pandas as pd
import numpy as np


def hierarchical_clustering(dist_mat, method='complete'):
    if method == 'complete':
        Z = complete(dist_mat)
    if method == 'single':
        Z = single(dist_mat)
    if method == 'average':
        Z = average(dist_mat)
    if method == 'ward':
        Z = ward(dist_mat)
    
#     fig = plt.figure(figsize=(16, 8))
#     dn = dendrogram(Z)
#     plt.title(f"Dendrogram for {method}-linkage with correlation distance")
#     plt.show()
    
    return Z


def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
              
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        print(distance)
        return "The strings are {} edits away".format(distance[row][col])
    
def dis_matrix_calc(seq):
    edit_distance = []

    flag = 0

    for i in range(0, len(seq)):
        edit_distance = []
        for j in range(0, len(seq)):

            edit_distance.append(levenshtein_ratio_and_distance(seq[i], seq[j]))

        if flag == 0:

            distance_matrix = pd.DataFrame([edit_distance[0:len(seq)]])
            flag = 1

        else:
            distance_matrix = distance_matrix.append([edit_distance[0:len(seq)]])

    distance_matrix = distance_matrix.reset_index(drop=True)
    
    return distance_matrix

