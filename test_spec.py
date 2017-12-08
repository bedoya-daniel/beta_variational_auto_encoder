#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:41:59 2017

@author: victorw
"""
#%%
import matplotlib.pyplot as plt
a1,a2,a3,a4 = DATASET.spectrograms[1], DATASET.spectrograms[96], DATASET.spectrograms[153],DATASET.spectrograms[199]

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(a1)
plt.subplot(2,2,2)
plt.imshow(a2)
plt.subplot(2,2,3)
plt.imshow(a3)
plt.subplot(2,2,4)
plt.imshow(a4)