#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:27:13 2020
For generating the flat transient profile map for different ISI bounds
@author: soukhind
"""

import numpy as np

#%% attnmap
tmap = np.zeros((20,20))
for l in np.arange(1,20,1):
    for u in np.arange(l,20,1):

        if l <= 7 and u <= 9:
            tmap[l,u] = tmap[l,u] + 1
            
        elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
              or (l == 8 and u >= 8 and u <= 9)
              or (l == 9 and u == 9)):
            tmap[l,u] = tmap[l,u] + 2
            
        elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
              or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
              or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
            tmap[l,u] = tmap[l,u] + 3
            
        elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
            or ((l == 10 or l == 11) and u >= 14 and u <= 20)
            or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
            or ((l == 14 or l == 15) and u >= 15 and u <= 20)
            or (l >= 16 and u >= 15 and u <= 20)):
            tmap[l,u] = tmap[l,u] + 4
            

            
#%% wmmap
            
tmap = np.zeros((20,20))
for l in np.arange(1,20,1):
    for u in np.arange(l,20,1):

        if ((l <= 9 and u <= 13)
            or (l <= 5)):
            tmap[l,u] = tmap[l,u] + 5      
            
        elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
              or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
              or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
            tmap[l,u] = tmap[l,u] + 6
            
        elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
            or ((l == 10 or l == 11) and u >= 14 and u <= 20)
            or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
            or ((l == 14 or l == 15) and u >= 15 and u <= 20)
            or (l >= 16 and u >= 15 and u <= 20)):
            tmap[l,u] = tmap[l,u] + 7
            
#%% 
        
for i in np.arange(0,len(d.temp)):
    if d.temp[i] == 1:
        print(i)
        print(dr = np.where(d.temp [i:] == 1)[0][0])