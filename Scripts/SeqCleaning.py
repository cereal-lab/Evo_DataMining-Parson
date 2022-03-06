#!/usr/bin/env python
# coding: utf-8

import re


### Sequence cleaning 


### process the string (Eliminate the characters)
def processedString(str1, removedChar):
    transStr1 = str1.maketrans('','', removedChar)
    str1 = str1.translate(transStr1)

    return str1

### keep how many first labels
### variable 'keep' xth first label
def processedString_2(str2, removedChar, keep):

    count = 1
    seq = ""
    for j in str2:
        if j == removedChar and count <= keep:
            count += 1
            seq += j
            continue

        if j == removedChar and count > keep:
            count += 1
            continue

        seq += j
        
    return seq

### keep the character according to the list
### variable 'keep' is the list with binary, e.g. = [0,1,0,1]
def processedString_3(str2, removedChar, keep):
    
    count = 0
    seq = ""
    
    for j in str2:        
        if count < len(keep):
            if j == removedChar and keep[count] == 1:
                count += 1
                seq += j
                continue
                

        if j == removedChar:
            count += 1
            continue
            
        seq += j

    return seq

### clean the distractor
def no_distrator(df_x):
    ###Removing the lowercase
    df_x['new_sequence'] = df_x['new_sequence'].apply(lambda x: re.sub('[a-z]', '', x))
    
    return df_x

# def clear_after_submission(df_x):
#     df_x['new_sequence'] = df_x['new_sequence'].apply(lambda x: x.split("S", 1)[0])
    
#     return df_x

### eliminate all x label in the seq
def clear_label(df_x, clearing):
    df_x['new_sequence'] = df_x['new_sequence'].apply(lambda x: processedString(x, clearing))
#     df_x['new_steps'] = df_x['new_sequence'].map(len)
    
    return df_x

### keep labels based on the conditions
def keep_label(df_x, chars, num1, num2):
    df_x['new_sequence'] = df_x['new_sequence'].apply(lambda x: processedString_3(x, chars[0], num1))
    df_x['new_sequence'] = df_x['new_sequence'].apply(lambda x: processedString_3(x, chars[1], num2))
    
    return df_x

### eliminate all parenthesis
def clear_parenthesis(df_x):
    df_x = df_x[df_x['new_sequence'].str.count('P')  == 4 ]
    df_x = df_x.reset_index(inplace = False, drop=True)
    
    return df_x

