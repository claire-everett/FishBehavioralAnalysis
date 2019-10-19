#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:58:45 2019

@author: Claire
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns; sns.set()
from scipy.ndimage.morphology import binary_dilation


def mydistance(pos_1,pos_2):
    '''
    Takes two position tuples in the form of (x,y) coordinates and returns the distance between two points
    '''
    x0,y0 = pos_1
    x1,y1 = pos_2
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)

    return dist

def nanarccos (floatfraction):
    con1 = ~np.isnan(floatfraction)
    
    if con1:
        cos = np.arccos(floatfraction)
        OPdeg = np.degrees(cos)
    
    else:
        OPdeg = np.nan
    
    return (OPdeg)

        
def vecnanarccos():
    
    A = np.frompyfunc(nanarccos, 1, 1)
    
    return A
    
 
def lawofcosines(line_1,line_2,line_3):
    '''
    Takes 3 series, and finds the angle made by line 1 and 2 using the law of cosine
    '''
 
    num = line_1**2 + line_2**2 - line_3**2
    denom = (line_1*line_2)*2
    floatnum = num.astype(float)
    floatdenom = denom.astype(float)
    floatfraction = floatnum/floatdenom
    OPdeg = vecnanarccos()(floatfraction)
    
    return OPdeg

def midpoint (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition pos_1: x-value object 1, pos_2: y-value object 1, pos_3: x-value object 2
    pos_4: y-value object 2
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2

    return (midpointx, midpointy)

def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    return (data['x'],data['y'])

def binarize(anglearrays):
    booleans = []
    for i in range(2):
        anglearray = anglearrays[i]
        inds = np.arange(len(anglearray))
        boolean = anglearray.apply(lambda x: 1 if x > 140 else 0).values
        boolean = binary_dilation(boolean, structure = np.ones(40,))
        booleans.append(boolean)
    
    product = booleans[0] * booleans[1]
    result = np.where(product)
    
    return (result)

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
    
    Operangle = lawofcosines(HROP,HLOP,RLOP)
    
    return Operangle

## Package up filtering steps. Just expedient for the moment, revise later
def auto_scoring_tracefilter(data,p0=20,p2=15):
    mydata = data.copy()
    boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
    for b in boi:
        for j in ['x','y']:
            xdifference = abs(mydata[b][j].diff())
            xdiff_check = xdifference > p0     
            mydata[b][j][xdiff_check] = np.nan
 
            origin_check = mydata[b][j] < p2
            mydata[origin_check] = np.nan

    return mydata


def getfiltereddata(h5_files):
    file_handle1 = h5_files[0]

    with pd.HDFStore(file_handle1,'r') as help1:
        data_auto1 = help1.get('df_with_missing')
        data_auto1.columns= data_auto1.columns.droplevel()
        data_auto1_filt = auto_scoring_tracefilter(data_auto1)
     
    
    return data_auto1_filt

binindex =[]
def binarizeOp(Operangle):
    

    boolean = Operangle.apply(lambda x: 1 if x > 65 else 0).values
#    boolean = binary_dilation(boolean, structure = np.ones(40,))
    
    binindex = np.where(boolean)[0]
#    for i,value in enumerate(boolean):
#        if value == 1:
#            binindex.append([i])
#            print('hey')
#        else:
#            binindex.append(np.nan)
#            print('bye')
#   
    return (binindex)
## End of object for loading and extracting operculum opening 
    
def uniform(indbin,LoopLength):
    A = len(indbin)
    B = LoopLength
    C = B - A
    end = np.zeros(C)
    new = np.concatenate((indbin, end))
    
    return(new)
    
home_dir = '.'#'/Users/Claire/Desktop/FishBehaviorAnalysis'
h5_files = sorted(glob(os.path.join(home_dir,'*.h5')))
print(h5_files)

## Packaged up some of the upload code. 
data_auto1_filt = getfiltereddata(h5_files)

Operangle = auto_scoring_get_opdeg(data_auto1_filt)

# split operangle into 1474 windows

LoopLength = 1474
NumLoops = int(int(len(Operangle))/LoopLength)
Looped = pd.DataFrame(np.reshape(Operangle.values[:(LoopLength*NumLoops)],(int(len(Operangle)/LoopLength), LoopLength))).T

# will return a dataframeof length 1473 and as many columns as there are full repeats 

# Now I'm going to estimate when the video would have been turned on

FirstThird = int(NumLoops/3)
SecondThird = int(FirstThird*2)
Middle = int((FirstThird+SecondThird)/2)
#
#for i in np.arange(FirstThird, FirstThird+10):
#    plt.plot(Looped[i])


## to make raster plot, binarize data, then record what indexes coincide with 1, 
    # make a list of those indeces, plot those

indbins = np.random.rand(0,LoopLength)
#
for i in np.arange(FirstThird,SecondThird):
    indbin = uniform(binarizeOp(Looped[i]), LoopLength)
    indbins = np.vstack([indbins, indbin])

plt.eventplot(indbins)
plt.savefig('rasterfirst.pdf')

## to calc the actual frame that corresponds
Loop = 20
WithinLoop = 0

CurrentFrame = FirstThird + Loop + WithinLoop

