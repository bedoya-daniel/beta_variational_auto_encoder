#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:50:59 2017

@author: victorw
"""
from numpy import *
import itertools as it

my_dict={'A':['D','E'],'B':['F','G','H'],'C':['I','J']}
allNames = sorted(my_dict)
combinations = it.product(*(my_dict[Name] for Name in allNames))
print(list(combinations))


my_dict = {
        'f0':  arange(100,1000,50),
        'snr': arange(1,2,0.01),
        'ps' : arange(0,5,1)
        }

allNames = sorted(my_dict)
blabla = (my_dict[Name] for Name in allNames)
combinations = it.product(blabla)
print(list(combinations))



#%%
f0 =  arange(100,1000,50)
snr= arange(1,2,0.01)
ps = arange(0,5,1)

myprod = it.product(f0, snr, ps)
dicto = {}

for p in combinations:
     a = array(list(p))
    
