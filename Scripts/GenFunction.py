#!/usr/bin/env python
# coding: utf-8

import Scripts.SeqCleaning as seqCleaning
import Scripts.GAFunctions as GAfunctions
import Scripts.StatFitness as statFitness

### General functions

### Convert a list of binary bits to a decimal number
def convert_to_decimal(binary_list):
    res = int("".join(str(x) for x in binary_list), 2)
    
    return int(res)

### Convert a decimal number a list of binary bits
def convert_to_binary(decimal_num, bits):
    res = format(decimal_num, "0{}b".format(bits))
    res = [int(x) for x in res]
    
    return res

### Obtain the summary
### Counting how many label x in each sequence
def count_label(df_x,char_x):
    counting = []
    for row in df_x['sequence']:
        counting.append(row.count(char_x))
        
    return counting

### Counting how many label x on average
def count_average_label(df_x,char_x):
    total_label = 0
    for row in df_x['sequence']:
        total_label = total_label + row.count(char_x)
        
    print(total_label/len(df.index))

### information of the label x
def label_info(df_x, char_x, num_x):
    labels = count_label(df_x, char_x)
    print(len([x for x in labels if x >= num_x]), "rows having or more than", num_x ,"label", char_x)
    print()
    
    labels_list = {x:labels.count(x) for x in sorted(labels)}
    accumulated_value = 0
    for key,value in labels_list.items():
        accumulated_value = accumulated_value + value
        print(key , ':' , value, '-', accumulated_value)   


def dataset_cleaning(df_x, l_mask, clear_bits):
    df_x['new_sequence'] = df_x['sequence']
    df_x = seqCleaning.clear_parenthesis(df_x) ### Clear the parenthesis
    df_x = seqCleaning.no_distrator(df_x) ### Clear all the distractors label
#     df_1 = clear_after_submission(df_1)

    ### Randomly keep the label 'R' and keep first 5 'T'...
    df_x = seqCleaning.keep_label(df_x, 'RT', clear_bits[:l_mask], clear_bits[l_mask:])
    
#    df_x = seqCleaning.keep_label(df_x, 'RT', clear_bits[:l_mask], 5)
    df_x = seqCleaning.clear_label(df_x, 'S')
    
    return df_x



def calc_score(df_x, score_cache, l_mask, n_clusters, clear_bits):
    
    n = convert_to_decimal(clear_bits)
    if n in score_cache:
        ss = score_cache.get(n)[0]
        aic = score_cache.get(n)[1]
        dis = score_cache.get(n)[2]
        rsq = score_cache.get(n)[3]
        clusters_count = score_cache.get(n)[4]

    else:
        df_y = dataset_cleaning(df_x, l_mask, clear_bits)

        _, ss, aic, dis, rsq, clusters_count = statFitness.fitness(df_y, n_clusters)

    return n, ss, aic, dis, rsq, clusters_count





# def calc_stat(df_x, score_cache, l_mask, clear_bits):
    
#     n = convert_to_decimal(clear_bits)
#     if n in score_cache:
#         s = score_cache.get(n)
        
#     else:
# #         df_y = df_x.copy()
#         df_y = dataset_cleaning(df_x, l_mask, clear_bits)

#         _, s = statFitness.fitness(df_y)

#     return n, s
