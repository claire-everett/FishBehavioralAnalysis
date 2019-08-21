# coding: utf-8

# In[ ]:


#Notebook to quantify the duration of time the fish spends
#following the other fish
#going to measure using the orientation of the fish and how well
# the orientation matches the line from the tip of the head to the tip of the head
#of the other fish

##actually I can do it by just measuring a point in the middle
# of the fish, the tip of th ehead, and the tip of the other fish's head
# then the closer that angle is to 180, the more the fish is "facing" the other fish


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.spatial import distance
import seaborn as sns; sns.set()
import math
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm
from tqdm import tqdm_notebook as tqdm
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


# ##### Functions

# In[24]:


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

def midpointx (pos_1, pos_2, pos_3, pos_4):
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

def gaze_tracking(fish1,fish2):
    """
    A function that takes in two dataframes corresponding to the two fish. Assymetric. Fish one is the one gazing, fish two is being gazed at. 
    
    Parameters: 
    Fish1: A dataframe of tracked points. 
    Fish2: A dataframe of tracked points. 

    Output:
    A vector of angles between 0 and 180 degrees (180 corresponds to directed gaze).
    """

    ## Get the midpoint of the gazing fish. 
    midx,midy = midpointx(fish1['B_rightoperculum']['x'], fish1['B_rightoperculum']['y'], fish1['E_leftoperculum']['y'], fish1['E_leftoperculum']['y'] )
    line1 = mydistance((midx, midy), (fish1['A_head']['x'], fish1['A_head']['y']))
    line2 = mydistance((fish1['A_head']['x'], fish1['A_head']['y']), (fish2['A_head']['x'], fish2['A_head']['y']))
    line3 = mydistance((fish2['A_head']['x'], fish2['A_head']['y']), (midx, midy))

    String = lawofcosines(line1, line2, line3)
    return String


def gaze_ethoplot(anglearrays,title,show = True,save = False):
    """
    Function to generate a plot of the gaze of both fish, and points when they are looking in the same direction. 

    Parameters: 
    anglearrays: a list of two array-likes: one for fish one, one for fish two. 
    title: a string of the title for the plot. 
    show: a boolean. True = plot, False = do not plot
    save: a boolean: True = save, False = do not save. 
    """
    colors = ['red','blue']
    labels = ['fish1 gaze', 'fish2 gaze']
    fig,ax = plt.subplots(2,1,sharex = True)
    for i in range(2):
        anglearray = anglearrays[i]
        inds = np.arange(len(anglearray))
        color = colors[i]
        ax[0].plot(anglearray,color = color,linewidth = 1,alpha = 0.5,label = labels[i])
        boolean = anglearray.apply(lambda x: 1 if x > 170 else 0) 
        [ax[1].axvline(x = j,alpha = 0.2,color = color) for j in inds if boolean[j] == 1]

    ax[0].set_ylabel('relative angle (degrees)')
    ax[1].set_ylabel('threshold crossing')
    ax[1].set_xlabel('time (frame)')
    ax[0].set_title(title)

    ax[0].legend(loc = 1)## in the upper right; faster than best. 
    if show == True:
        plt.show()

####################################

    ## ##### Create Midpoint of Operculi

    ## In[25]:


    #data_auto1['Midpointx'], data_auto1['Midpointy'] = midpointx(data_auto1['B_rightoperculum']['x'], data_auto1['B_rightoperculum']['y'], data_auto1['E_leftoperculum']['y'], data_auto1['E_leftoperculum']['y'] )


    ## ##### Measure the String Between Fish1 Orientation and Fish2

    ## In[36]:


    ##line 1: between midpoint of op1 and tip of fish head 1
    ## line 2: between tip of fish head 1 and 2
    ## line 3: between tip of fish head 2 and midpoint of op1

    #line1 = mydistance((data_auto1['Midpointx'], data_auto1['Midpointy']), (data_auto1['A_head']['x'], data_auto1['A_head']['y']))
    #line2 = mydistance((data_auto1['A_head']['x'], data_auto1['A_head']['y']), (data_auto2['A_head']['x'], data_auto2['A_head']['y']))
    #line3 = mydistance((data_auto2['A_head']['x'], data_auto2['A_head']['y']), (data_auto1['Midpointx'], data_auto1['Midpointy']))

    #String = lawofcosines(line1, line2, line3)
    #funcstring = gaze_tracking(data_auto2,data_auto1) 
    #plt.plot(funcstring)
    #plt.show()


    ## In[49]:


    #StraightString = String.apply(lambda x: 1 if x > 170 else 0)


    ## In[50]:


    #StraightString.sum()


    # In[53]:


    #plt.plot(String)
if __name__ == "__main__":
    # ##### Load Data

    # In[4]:


    h5_dir = '.'#'/Users/Claire/Desktop/Test'
    h5_files = glob(os.path.join(h5_dir,'*.h5'))
    print(h5_files)


    # In[6]:

    file_handle1 = h5_files[1]

    with pd.HDFStore(file_handle1,'r') as help1:
        data_auto1 = help1.get('df_with_missing')
        data_auto1.columns= data_auto1.columns.droplevel()

    file_handle2 = h5_files[0]

    with pd.HDFStore(file_handle2,'r') as help2:
        data_auto2 = help2.get('df_with_missing')
        data_auto2.columns= data_auto2.columns.droplevel()


    angle1 = gaze_tracking(data_auto1,data_auto2)
    angle2 = gaze_tracking(data_auto2,data_auto1)

    gaze_ethoplot([angle1,angle2],'test',show = True)


    ### Plot points where this fish is looking at the other: 
    #print(len(StraightString))
    #look_inds = np.arange(len(StraightString))
    #print(look_inds)
    #[plt.axvline(x = i, alpha =0.2,color="red") for i in look_inds if StraightString[i] == 1]
    #plt.axvline(x = 89514,color = 'black')
    #plt.show()

