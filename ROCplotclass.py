#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:20:00 2019

@author: Claire
"""
''' 
this is to translate the ROC plots to spyder and to possibly add to a new class'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.spatial import distance
import seaborn as sns; sns.set()


def mydistance(pos_1,pos_2):
    '''
    Takes two position tuples in the form of (x,y) coordinates and returns the distance between two points
    '''
    x0,y0 = pos_1
    x1,y1 = pos_2    
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    return dist

def lawofcosines(line_1,line_2,line_3):
    ''' 
    Takes 3 series, and finds the angle made by line 1 and 2 using the law of cosine
    '''
    num = line_1**2 + line_2**2 - line_3**2
    denom = (line_1*line_2)*2
    floatnum = num.astype(float)
    floatdenom = denom.astype(float)
    floatfraction = floatnum/floatdenom
    cos = np.arccos(floatfraction)
    OPdeg = np.degrees(cos)
 
    return OPdeg

def speed (Xcords, Ycords, fps):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''

    distx = Xcords.diff()*.02 #converts distance to mm based on resolution/FOV
    disty = Ycords.diff()*.02 #converts distance to mm based on resolution/FOV
    TotalDist = np.sqrt(distx**2 + disty**2)
    Speed = TotalDist / (1/fps) #converts to seconds
    
    return Speed

def ThreeDspeed (Xcords, Ycords, Zcords, fps):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''

    distx = Xcords.diff()*.02 #converts distance to mm based on resolution/FOV
    disty = Ycords.diff()*.02 #converts distance to mm based on resolution/FOV
    distz = Zcords.diff()*.02 #converts distance to mm based on resolution/FOV
    TotalDist = np.sqrt(distx**2 + disty**2 + distz**2)
    Speed = TotalDist / (1/fps) #converts to seconds
    
    return Speed

def myvelocity (xcoords, ycoords):
    xdiff = []
    for i,value in enumerate (xcoords):
        if i < len(xcoords)-1:
            x1 = xcoords.iloc[i]
            x2 = xcoords.iloc[i + 1]
            xdifference = (x2 - x1)
            xdiff.append(xdifference)
        else:
            xdiff.append(np.nan)

    ydiff = []
    for i,value in enumerate (ycoords):
        if i < len(ycoords)-1:
            y1 = ycoords.iloc[i]
            y2 = ycoords.iloc[i + 1]
            ydifference = (y2 - y1)
            ydiff.append(ydifference)
        else:
            ydiff.append(np.nan)
    
    return (xdiff, ydiff)

def midpointx (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2
    
    return (midpointx, midpointy)


# ##### Defining Filtering and Thresholding Functions

# In[23]:


## Now we will write a function to output the manual scoring directly from the excel file: 
def manual_scoring(data_manual,data_auto,crop0 = 0,crop1= -1):
    '''
    A function that takes manually scored data and converts it to a binary array. 
    
    Parameters: 
    data_manual: manual scored data, read in from an excel file
    data_auto: automatically scored data, just used to establish how long the session is. 
    
    Returns: 
    pandas array: binary array of open/closed scoring
    '''
    Manual = pd.DataFrame(0, index=np.arange(len(data_auto)), columns = ['OpOpen'])
    reference = data_manual.index
    data_manual['Stop']
    for i in reference:
        Manual[data_manual['Start'][i]:data_manual['Stop'][i]] = 1
    return Manual['OpOpen'][crop0:crop1]


# In[24]:


## Now we will start writing functions to output the result of analyzing the behavioral traces to give automatic scoring: 
def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    return (data['x'],data['y'])

def auto_scoring_get_opdeg(data_auto):
    '''
    Function to automatically score operculum as open or closed based on threshold parameters. 
    
    Parameters: 
    data_auto: traces of behavior collected as a pandas array. 
    thresh_param0: lower threshold for operculum angle
    thresh_param1: upper threshold for operculum angle
    
    Returns:
    pandas array: binary array of open/closed scoring
    '''
    # First collect all parts of interest:
    poi = ['A_head','B_rightoperculum','E_leftoperculum']
    HROP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[1]]))
    HLOP = mydistance(coords(data_auto[poi[0]]),coords(data_auto[poi[2]]))
    RLOP = mydistance(coords(data_auto[poi[1]]),coords(data_auto[poi[2]]))

    return lawofcosines(HROP,HLOP,RLOP)

## Additional parameters for smoothing could be taken
def auto_scoring_smooth_opdeg(opdeg):
    #Smoothes OPdeg
    x=range(0,len(opdeg),1)
    w = np.isnan(opdeg)
    opdeg[w] = 0.
    u_p=UnivariateSpline(x,opdeg,w = ~w)
#     d = {'xcoord': range(0,len(opdeg),1), 'raw': opdeg,'smoothed':u_p}
    return pd.Series(u_p)

## Filters by width of detected events. 
def auto_scoring_widthfilter(binary_scores,widththresh = 30):

    fst = binary_scores.index[binary_scores & ~ binary_scores.shift(1).fillna(False).astype(bool)]
    lst = binary_scores.index[binary_scores & ~ binary_scores.shift(-1).fillna(False).astype(bool)]
    
    print(len(fst))
    if len(fst) < 1:
        width_filtered = pd.DataFrame(0, index=np.arange(len(binary_scores)), columns = ['OpOpen'])
    
    if len(fst) > 0:
        intv = pd.DataFrame([(i, j) for i, j in zip(fst, lst) if j > i + 10]) ## 10 is also a parameter..
        intv.columns=['start', 'end']
        intv['width'] = intv['end']-intv['start']
        intv = intv.loc[intv['width'] > widththresh]
        intv['new_col'] = range(0, len(intv))

        reference = intv.index
        width_filtered = pd.DataFrame(0, index=np.arange(len(binary_scores)), columns = ['OpOpen'])
        for i in reference:
            width_filtered[intv['start'][i]:intv['end'][i]] = 1

        return width_filtered

def auto_scoring_tracefilter(data,p0=20,p1=250,p2=15,p3=70,p4=200):
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
    mydata['bodylength'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[3]]))
    mydata['Operwidth'] = mydistance(coords(mydata[boi[4]]),coords(mydata[boi[1]]))
    mydata['HeadROperwidth'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[1]]))
    mydata['HeadLOperwidth'] = mydistance(coords(mydata[boi[0]]),coords(mydata[boi[4]]))
    mydata['TailtipROperwidth'] = mydistance(coords(mydata[boi[3]]),coords(mydata[boi[1]]))
    mydata['TailtipLOperwidth'] = mydistance(coords(mydata[boi[3]]),coords(mydata[boi[4]]))
    mydata['TailbaseLOperwidth'] = mydistance(coords(mydata[boi[2]]),coords(mydata[boi[4]]))
    mydata['TailbaseROperwidth'] = mydistance(coords(mydata[boi[2]]),coords(mydata[boi[1]]))

    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
    #         print (xdiff_check.loc[xdiff_check == True])
            mydata[xdiff_check] = np.nan
    #         print (mydata.loc[np.isnan(mydata['A_head']['x'])])

            bodylength_check = mydata['bodylength'] > p1
            mydata[bodylength_check] = np.nan

            origin_check = mydata[b][j] < p2
            mydata[origin_check] = np.nan

            Operwidth_check = mydata['Operwidth'] > p3
            mydata[Operwidth_check] = np.nan

            HeadROperwidth_check = mydata['HeadROperwidth'] > p3
            mydata[HeadROperwidth_check] = np.nan

            HeadLOperwidth_check = mydata['HeadLOperwidth'] > p3
            mydata[HeadLOperwidth_check] = np.nan

            TTL_check = mydata['TailtipLOperwidth'] > p4
            mydata[TTL_check] = np.nan

            TTR_check = mydata['TailtipROperwidth'] > p4
            mydata[TTR_check] = np.nan
    return mydata

##############################################################################
def auto_scoring_TS1(data_auto,thresh_param0 = 70,thresh_param1 = 180):
    '''
    Function to automatically score operculum as open or closed based on threshold parameters. 
    
    Parameters: 
    data_auto: traces of behavior collected as a pandas array. 
    thresh_param0: lower threshold for operculum angle
    thresh_param1: upper threshold for operculum angle
    
    Returns:
    pandas array: binary array of open/closed scoring
    '''
    degree = auto_scoring_get_opdeg(data_auto)
    return degree.apply(lambda x: 1 if thresh_param0 < x < thresh_param1 else 0)
##############################################################################

##############################
def auto_scoring_M2(data_auto,thresh_param0 = 70,thresh_param1 = 180,thresh_param3 = 30):
    
    raw_out = auto_scoring_TS1(data_auto,thresh_param0,thresh_param1)
    
    widthfilter_out = auto_scoring_widthfilter(raw_out,widththresh = 30)
    return widthfilter_out

##############################
#### 10 parameters to play with. 
    
h5_dir = '/Users/Claire/Desktop/Python'
h5_files = glob(os.path.join(h5_dir,'*.h5'))
h5_files


Fish1aut = angle1.apply(lambda x: 1 if x > 140 else 0).values
Fish1Man = manual_scoring(data_manual, data_auto2[87696: 92536])


sns.heatmap(Compare)
