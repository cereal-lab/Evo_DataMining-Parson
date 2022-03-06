#!/usr/bin/env python
# coding: utf-8

# In[1]:


from csv import reader
import pandas as pd
import numpy as np

import random
import sys

import os
import shutil

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import time

import multiprocessing
from functools import partial

# genetic algorithm search of the one max optimization problem
from numpy.random import randint
# from numpy.random import rand


import Scripts.LevenClustering as levenClustering
import Scripts.SeqCleaning as seqCleaning
import Scripts.GenFunction as genFunction
import Scripts.GAFunctions as GAfunctions


# In[2]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

output = "Puzzles_Output"
# file = ['00 Result/SelectionF15/puzzle_1.csv', '00 Result/SelectionS16/puzzle_1.csv']
file = ['Puzzles_CSV/Puzzle_1.csv']


# In[3]:


### GA parameters
# n_pop = int(sys.argv[1]) # define the population size
n_pop = 30
print("population: ", n_pop)

# n_iter = int(sys.argv[2]) # define the total iterations
n_iter = 50
print("generation: ", n_iter)

n_bits = 58 # bits
label_mask = 44

r_cross = 0.5 # crossover rate
r_mut = 1.0 / float(n_bits) # mutation rate


# In[4]:


col_names = ['File Path', 'sequence', 'steps', 'time_spent']
df = pd.DataFrame(columns = col_names)

file_path = []
sequence = []
steps = []
time_spent = []

pop_cache = {}

if __name__ == "__main__":


    for f in file:

        with open(f, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)

            if header != None:
                for row in csv_reader:
                    file_path.append(row[0])
                    sequence.append(row[6])
                    steps.append(row[3])
                    time_spent.append(row[5])

    df['File Path'] = file_path
    df['sequence'] = sequence
    df['steps'] = steps
    df['time_spent'] = time_spent

    print("Total Rows for original: ", len(df.index))
    
    col_names = ['Generation', 'Population', 'Combination', 'R', 'T', 'Silhouete']
    df_result = pd.DataFrame(columns = col_names)
    
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # assign the first combination as the best
    best, best_eval = 0, 0

    ts2_before = time.time()

    df_1 = df.copy()
    
    for gen in range(n_iter):

        scores = []

        ts_before = time.time()
        
        ### cache
#         for i in range(len(pop)):
#             num, score = genFunction.calc_score(df_1, pop[i], pop_cache, label_mask)

#             scores.append(score)
#             pop_cache[num] = score
            
            
        pool = multiprocessing.Pool(4)
        combined_args = partial(genFunction.calc_score, df_1, pop_cache, label_mask)
        result = pool.map(func=combined_args, iterable=pop)
        pool.close()
        pool.join()
        
        pop = []
        scores = []
        
        for v in result:
            pop_cache[v[0]] =  v[1]
            
            pop.append(genFunction.convert_to_binary(v[0], n_bits))
            scores.append(v[1])


#         for i in range(len(pop)):
#             print(pop[i], ":", scores[i])
            
#         print(result)
            
#         for key,value in pop_cache.items():
#             print(key, "::", value)
            
            
        ### sort the list and pick the best 100 combination into the next population
        pop = [x for _, x in sorted(zip(scores, pop), reverse = True)]
        scores = sorted(scores, reverse = True)
        pop = pop[:n_pop]
        scores = scores[:n_pop]

        for i in range(n_pop):
        ### Get the best combination and score
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]


            
        ts_after = time.time()
        ts = ts_after - ts_before

        print(f'Gen: {gen}, R = {best[:label_mask]}, T = {best[label_mask:]}, Silhouete Score(n=3): {round(best_eval,4)}, Time Taken: {round(ts, 2)}s')

        ### Storing the result-----------------------
        df_result.loc[gen, 'Generation'] = gen
        df_result.loc[gen, 'Population'] = len(pop)
        df_result.loc[gen, 'Combination'] = best
        df_result.loc[gen, 'R'] = best[:label_mask]
        df_result.loc[gen, 'T'] = best[label_mask:]
        df_result.loc[gen, 'Silhouete'] = best_eval

        ### GA Operations----------------------------
        # select parents
        selected = [GAfunctions.selection(pop, scores) for _ in range(n_pop)]  

        children = []

        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]

            for c in GAfunctions.crossover(p1, p2, r_cross):

                # mutation
                GAfunctions.mutation(c, r_mut)

                # store for next generation
                children.append(c)

        # replace population
        pop = pop + children
        
        ### --------------------------------------------
        
    ts2_after = time.time()
    ts2 = ts2_after - ts2_before
    
    df_final = genFunction.dataset_cleaning(df_1, label_mask, best)

    print("Done--------------------------------------", ts2)

    if os.path.exists(output):
        shutil.rmtree(output)

    if not os.path.exists(output):
        os.makedirs(output)


    df_result.plot(kind='line',x='Generation',y='Silhouete',color='red')
    plt.title("Population: {}, R bits: {}, T bits: {}".format(n_pop, label_mask, (n_bits-label_mask)))
    plt.savefig('{}/output.png'.format(output))
    plt.show()


    df_result.to_csv('{}/out.csv'.format(output), index=False)  
    df_final.to_csv('{}/final_output.csv'.format(output), index=False)  
    
    
    
    


# In[5]:


# label_info(df,'R',4)


# In[6]:


print(pop_cache)


# In[ ]:




