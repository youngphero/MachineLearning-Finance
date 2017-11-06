#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:07:08 2017

@author: ahmedthabit
"""


import pandas as pd
import numpy as np

weekly_price=pd.read_excel('eurusdweekly_23_9_17.xlsx')



#DROP OPEN AND CLOSE

weekly_price.drop('Open', axis=1, inplace=True)

weekly_price.drop('Close', axis=1, inplace=True)


 
weekly_price['Diff_H']=weekly_price['High'].diff(-1)


weekly_price['Diff_L']=weekly_price['Low'].diff(-1)
#weekly_price['Diff_L']=weekly_price.Diff_L.shift(-1)

weekly_price.to_csv('eurweekly.csv')

weekly=pd.read_csv('eurweekly.csv')

#df['elderly'] = np.where(df['age']>=50, 'yes', 'no')

weekly['Diff_H']=np.where(weekly['Diff_H']>0,'HH','LH')

weekly['Diff_L']=np.where(weekly['Diff_L']>0,'HL','LL')

weekly.to_csv('weekly_update.csv')



1-  LHLL
2-  HHHL
3- hhhl



