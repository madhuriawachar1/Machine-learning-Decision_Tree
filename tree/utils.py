import numpy as np
import pandas as pd


def entropy(Y):
    classes = dict()
    for i in range(Y.size):
        if(Y.iat[i] in classes):
            classes[Y.iat[i]] += 1
        else:
            classes[Y.iat[i]] = 1
    
    entropy = 0
    for i in classes.keys():
        p_i = classes[i]/Y.size
        entropy -= (p_i*np.log2(p_i))
    
    return entropy



def gini_index(Y):
    classes = dict()
    for i in range(Y.size):
        if(Y.iat[i] in classes):
            classes[Y.iat[i]] += 1
        else:
            classes[Y.iat[i]] = 1
    
    gini = 1

    for i in classes.keys():
        p_i = classes[i]/Y.size
        gini -= np.square(p_i)
    
    return gini

    


def information_gain(Y, attr):
    assert(attr.size==Y.size)

    classes_attr = dict()
    for i in range(attr.size):
        if(attr.iat[i] in classes_attr):
            classes_attr[attr.iat[i]].append(Y.iat[i])
        else:
            classes_attr[attr.iat[i]] = [Y.iat[i]]
    
    gain = entropy(Y)

    for i in classes_attr.keys():
        p_i = len(classes_attr[i])/attr.size
        gain -= (p_i*entropy(pd.Series(data=classes_attr[i])))
    
    return gain