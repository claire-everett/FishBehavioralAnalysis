#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.spatial import distance
import seaborn as sns; sns.set()
import math
import moviepy
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


# #### Files / paths

# In[3]:


h5_dir = '/Users/Claire/Desktop/Python'
h5_files = glob(os.path.join(h5_dir,'*.h5'))
h5_files


# #### Load data from `h5` file into pandas dataframe

# In[4]:


file_handle = h5_files[0]
with pd.HDFStore(file_handle,'r') as help2:
    data = help2.get('df_with_missing')


# #### Import modules

# In[5]:


data


# ##### list columns

# In[6]:


data.columns


# 
# 
# #### Clean up dataframe

# In[7]:


data.columns= data.columns.droplevel()
data.head()


# 
# ##### Remove # Below Threshold

# In[8]:


# bodyparts_of_interest = ['A_head','B_rightoperculum','E_leftoperculum']

# for b in bodyparts_of_interest:
#     for j in ['x','y']:
#         new_res = []
#         for i in data[b][j]:
#             if i < 15:
#                 new_res.append(np.nan)
#             else:
#                 new_res.append(i)
#         data[b][j] = new_res
# data.head()


# 
# 
# 
# ##### Smooth Data

# ### Guassian Sliding Window Adjustment

# ###### Finds mean of standard deviations of series within dataframe

# In[9]:


datasmooth = data.rolling(15, win_type='gaussian').mean(std= 1)
# datasmooth2 = data.rolling(10, win_type='triang').mean()
# datasmooth


# ##### Compare Original and Smoothed Data

# In[10]:


plt.figure(1, figsize=(20,10))
plt.subplot(2,2,1)
plt.plot(data['A_head']['x'][40000:50000])
# plt.subplot(2,2,2)
# plt.plot(datasmooth2['A_head']['x'][40000:50000])
plt.subplot(2,2,3)
plt.plot(datasmooth['A_head']['x'][40000:50000])


# 
# 
# 
# 
# 
# ## Change Range Depending on Video

# In[11]:


# datasmooth = datasmooth[:120250]


# ##### Distance between Head and Operculum

# Define functions

# In[12]:


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
    cos = np.cos(floatfraction)
    OPdeg = np.degrees(cos)
    
    return OPdeg


# ##### distance between Head and Right Operculum

# In[13]:


HROP = mydistance((datasmooth['A_head']['x'],datasmooth['A_head']['y']),(datasmooth['B_rightoperculum']['x'],datasmooth['B_rightoperculum']['y']))
HROP.tail()


# ##### Distance between Head and Left Operculum

# In[14]:


HLOP = mydistance((datasmooth['A_head']['x'],datasmooth['A_head']['y']),(datasmooth['E_leftoperculum']['x'],datasmooth['E_leftoperculum']['y']))
HLOP.tail()


# 
# ##### Distance between Right and Left Operculum

# In[15]:


RLOP = mydistance((datasmooth['B_rightoperculum']['x'],datasmooth['B_rightoperculum']['y']),(datasmooth['E_leftoperculum']['x'],datasmooth['E_leftoperculum']['y']))
RLOP.tail()


# ##### Law of Cosines to find Operculum Opening Angle

# In[17]:


datasmooth = data.rolling(10, win_type='gaussian').mean(std= 11)

#spline function

datasmooth['OPdeg'] = lawofcosines(HROP,HLOP,RLOP)


# ##### Check Dimensions of OPdeg

# In[18]:


plt.figure(figsize=(13,3))
plt.plot(datasmooth['OPdeg'], linewidth=0.5)
plt.title('Angle over Time')


# In[19]:


plt.figure(figsize=(13,3))
A = sns.kdeplot(datasmooth['OPdeg'], bw=0.25)
plt.title('Frequency of Angles')

A = plt.savefig('BimodalOPdeg.pdf')


# ##### For every value in OPdeg above 55, add a 1 to new column 'OPopen'

# In[ ]:





# In[26]:


datasmooth['OPopen'] = datasmooth['OPdeg'].apply(lambda x: 1 if x > 56 else 0)


# In[27]:


frametosec = 40000/25
frametomin = frametosec/60
frametomin


# In[28]:


mintoframe = ((1*60)+30)*25
mintoframe


# ##### Find how many frames had an open operculum

# In[29]:


NumPerOpen = (datasmooth['OPopen'][:72000] == 1).sum()
NumPerOpen


# ##### Find how many total frames 

# In[30]:


DenomPerOpen = len(datasmooth['OPopen'][:72000])
DenomPerOpen


# ##### % of Operculum Opened Frames/Total # Frames

# In[31]:


PerOpen = NumPerOpen/DenomPerOpen
PerOpen


# ##### 2D Speed

# In[ ]:


def speed (Xcords, Ycords, fps):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''

    distx = Xcords.diff()*.02 #converts distance to mm based on resolution/FOV
    disty = Ycords.diff()*.02 #converts distance to mm based on resolution/FOV
    TotalDist = np.sqrt(distx**2 + disty**2)
    Speed = TotalDist / (1/fps) #converts to seconds
    
    return Speed


# In[ ]:


datasmooth['2DSpeed(mm/s)'] = speed(datasmooth['A_head']['x'],datasmooth['A_head']['y'],90)


# In[14]:


plt.plot(datasmooth['2DSpeed(mm/s)'])
datasmooth


# In[15]:


plt.plot(datasmooth['OPopen'])


# In[16]:


AvgSpeed = datasmooth['2DSpeed(mm/s)'].mean()
AvgSpeed


# ##### Average Speed when Operculum is Open

# In[503]:


OPdata = datasmooth.loc[datasmooth['OPopen'] == 1.0]


# In[507]:


OPdata.head()


# ##### Avg Speed During Open Operculum

# In[508]:


OPAvgSpeed = OPdata['2DSpeed(mm/s)'].mean()
OPAvgSpeed


# ##### Joint Plot of Operculum angle and 2D Velocity

# ##### 3D Speed

# In[509]:


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


# In[510]:


datasmooth['3DSpeed (mm/s)'] = ThreeDspeed(datasmooth['A_head']['x'],datasmooth['A_head']['y'],datasmooth['B_rightoperculum']['x'], 90)
datasmooth.tail()


# ##### 2DVelocity 

# In[511]:


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


# In[512]:


# (A,B) = myvelocity(datasmooth['head']['x'], datasmooth['head']['y'])
datasmooth['xvelocity'] = datasmooth['A_head']['x'].diff()
datasmooth['yvelocity'] = datasmooth['A_head']['y'].diff()


# ##### 2D Angles Between Vectors

# In[513]:


xvel = datasmooth['xvelocity']
yvel = datasmooth['yvelocity']
Inter = np.vstack((xvel,yvel))

Inter = Inter.T
datasmooth['vectornorm'] = np.linalg.norm(Inter, axis = 1, keepdims = True)


# In[526]:


UnitV = Inter/np.linalg.norm(Inter, axis = 1, keepdims = True)

Angle = []
for i,value in enumerate(UnitV):
    if i < len(UnitV)-1:
        v1 = UnitV[np.array([i])]
        v2 = UnitV[np.array([i+1])]
        dotted = np.einsum('ik,ik->i', v1, v2)
        A = np.degrees(np.arccos(dotted))
        Angle.append(A)
    else:
        Angle.append(np.nan)
# print(Angle)
    
datasmooth['AngVect'] = Angle
datasmooth
plt.plot(datasmooth['AngVect'])


# In[515]:


# headdist = datasmooth['head']['x'].diff()
# test1 = datasmooth[np.abs(test)>100]
# headOP1 = distance((datasmooth['head']['x'],datasmooth['head']['y']), (datasmooth['Left Oper']['x'], datasmooth['Left Oper']['y']))
# test2 = datasmooth[np.abs(headOP1)>300]
# headOP2 = distance((datasmooth['head']['x'],datasmooth['head']['y']), (datasmooth['Right Oper']['x'], datasmooth['Right Oper']['y']))
# datasmooth = datasmooth[np.abs(headOP2)>300].nan


# ##### Visualization

# In[516]:


plt.plot(datasmooth['xvelocity'],datasmooth['yvelocity'])
sns.jointplot(datasmooth['vectornorm'], datasmooth['OPdeg'], alpha = 0.1,xlim = (0,10))
sns.jointplot(datasmooth['2DSpeed(mm/s)'], datasmooth['OPdeg'], alpha = .1)


# In[518]:


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(datasmooth['A_head']['x'], datasmooth['A_head']['y'], datasmooth['B_rightoperculum']['x'], c='skyblue', s=3)
# ax.view_init(2000, 2000)
# plt.title('3D Path')
# plt.show()


# In[519]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(datasmooth['xvelocity'], datasmooth['yvelocity'], datasmooth['OPdeg'], c='skyblue', s=3)
ax.view_init(2000, 2000)
plt.title('Velocity vs. OpDeg')
plt.show()


# In[520]:


framenum=10
frametomin = framenum/40
min=2
sec = 50
mintoframe = ((min*60)+sec)*40
mintoframe


# In[32]:


plt.plot(datasmooth['A_head']['x'][:72020], datasmooth['A_head']['y'][:72020], linestyle='-', markersize=0.7)
plt.plot(OPdatasmooth['A_head']['x'], OPdatasmooth['A_head']['y'], linestyle='-', markersize=0.7)
# plt.plot( data['A_head']['x'][2400:-1], data['A_head']['y'][2400:-1], linestyle='-', markersize=0.7, alpha = .5)
# plt.title('2D Path')
plt.show()


# In[522]:


# A = sns.heatmap(datasmooth['A_head'][['x','y']])
# plt.imshow(datasmooth['A_head'], cmap='hot', interpolation='nearest')


# In[523]:


n = 1000
T=np.linspace(0,1,np.size(datasmooth['A_head']['x']))**2
fig = plt.figure()
ax = fig.add_subplot(111)

s = 10 # Segment length
for i in range(0,n-s,s):
    ax.plot(datasmooth['A_head']['x'][i:i+s+1],datasmooth['A_head']['y'][i:i+s+1],color=(0.0,0.5,T[i]))


# ##### Head Direction

# ###### Defining Midpoint of Operculum

# In[ ]:


def midpointx (pos_1, pos_2, pos_3, pos_4):
    '''
    give definition
    '''
    midpointx = (pos_1 + pos_3)/2
    midpointy = (pos_2 + pos_4)/2
    
    return (midpointx, midpointy)


# In[ ]:


# # (datasmooth['OPMidpx'],datasmooth['OPMidpy'])  = midpointx(datasmooth['B_rightoperculum']['x'],datasmooth['B_rightoperculum']['y'], datasmooth['E_leftoperculum']['x'], datasmooth['E_leftoperculum']['y'])
# # datasmooth['OPMidpx'] = (datasmooth['Right Oper']['x']+datasmooth['Left Oper']['x'])/2
# # datasmooth['OPMidpy'] = (datasmooth['Right Oper']['y']+datasmooth['Left Oper']['y'])/2
# A = datasmooth['C_tailbase']['x']
# B = datasmooth['C_tailbase']['y']
# C = datasmooth['A_head']['x']
# D = datasmooth['A_head']['y']
# E = datasmooth['C_tailbase']['x']+100
# F = datasmooth['C_tailbase']['y']

# HT = mydistance((A,B),(C,D))
# TF = mydistance((A,B),(E,B))
# FH = mydistance((E,D),(C,D))

# # datasmooth['eastpointx'] = datasmooth['A_head']['x']+10
# # datasmooth['eastpointy'] = datasmooth['A_head']['y']
# # datasmooth

# Z = HT**2
# Y = TF**2
# X = FH**2
# lol = Z + Y - X
# HAH = 2* HT * TF
# Frac = lol/HAH
# HE = np.degrees(np.arccos(Frac))
# HE


# plt.plot(HE[6500:7200])
# plt.plot(datasmooth['OPdeg'][6500:7200])
# # plt.plot(datasmooth['A_head']['x'][6500:7200])
# plt.figure(figsize=(13,3))
# sns.kdeplot(HE, bw=0.25)

# datasmooth['HeadDirect'] = HE


# In[ ]:


(datasmooth['OPMidpx'],datasmooth['OPMidpy'])  = midpointx(datasmooth['B_rightoperculum']['x'],datasmooth['B_rightoperculum']['y'], datasmooth['E_leftoperculum']['x'], datasmooth['E_leftoperculum']['y'])
# datasmooth['OPMidpx'] = (datasmooth['Right Oper']['x']+datasmooth['Left Oper']['x'])/2
# datasmooth['OPMidpy'] = (datasmooth['Right Oper']['y']+datasmooth['Left Oper']['y'])/2
A = datasmooth['OPMidpx']
B = datasmooth['OPMidpy']
C = datasmooth['A_head']['x']
D = datasmooth['A_head']['y']
E = datasmooth['OPMidpx']+100
F = datasmooth['OPMidpy']

HT = mydistance((A,B),(C,D))
TF = mydistance((A,B),(E,B))
FH = mydistance((E,D),(C,D))

# datasmooth['eastpointx'] = datasmooth['A_head']['x']+10
# datasmooth['eastpointy'] = datasmooth['A_head']['y']
# datasmooth

Z = HT**2
Y = TF**2
X = FH**2
lol = Z + Y - X
HAH = 2* HT * TF
Frac = lol/HAH
HE = np.degrees(np.arccos(Frac))
HE


plt.plot(HE)
plt.plot(datasmooth['OPdeg'])
# plt.plot(datasmooth['A_head']['x'][6500:7200])
plt.figure(figsize=(13,3))
sns.kdeplot(HE, bw=0.25)

datasmooth['HeadDirect'] = HE


# ## Time Spent Near Glass, Looking at Glass:

# In[ ]:


#do the opdeg thing but two conidtions
# when the head x is greater than 400 and the head direction was between 0 and 25- marked as near and looking

datasmooth['near'] = (datasmooth['A_head']['x'].apply(lambda x: 1 if 450 < x < 475 else 0)) 
datasmooth['facing'] = datasmooth['HeadDirect'].apply(lambda x: 1 if 0 < x < 45 else 0)
nf = datasmooth['near'] + datasmooth ['facing']
# datasmooth['nearfacing'] = datasmooth.apply(lambda datasmooth: datasmooth.near + datasmooth.facing, axis=1)
# datasmooth
# FacingNum = datasmooth['union'].sum()
# Total = len(datasmooth['union'])
# Pertimefacing = FacingNum/Total
# Pertimefacing
# datasmooth

# s1 = pd.Series([4,5,6,20,42])
# s2 = pd.Series([1,2,3,5,42])

# s1[s1.isin(s2)]
A = nf[nf == 2].sum()
B = A/2
C = len(datasmooth['facing'])
Pertimefacing = B/C
Pertimefacing


# ## Time Flaring Near and Facing Glass and Approach

# In[ ]:


datasmooth['near'] = (datasmooth['A_head']['x'].apply(lambda x: 1 if 450 < x < 475 else 0)) 
datasmooth['facing'] = datasmooth['HeadDirect'].apply(lambda x: 1 if 0 < x < 45 else 0)

nf = datasmooth['near'] + datasmooth ['facing'] + datasmooth['OPopen']

A = nf[nf == 3].sum()
B = A/3
C = len(datasmooth['facing'])
Pertimefacing = B/C
Pertimefacing


# ### Approach w/ flaring

# ##### depends on which side the screen is on, be careful - take note of 2D plot

# In[ ]:


approach = []
headx = datasmooth['A_head']['x']
for i,value in enumerate (headx):
    if i < len(headx)-2:
        x1 = headx.iloc[i]
        x2 = headx.iloc[i + 1]
        x3 = headx.iloc[i + 2]
        if x3 > x2 > x1:
            approach.append(1)
        else:
            approach.append(0)
approach.append(np.nan)
approach.append(np.nan)
datasmooth['approach'] = approach

nf = datasmooth['approach'] + datasmooth ['facing'] + datasmooth['OPopen']

A = nf[nf == 3].sum()
B = A/3
C = len(datasmooth['facing'])
Pertimeapproaching = B/C
Pertimeapproaching


# In[ ]:


approach = []
headx = datasmooth['A_head']['x']
heady = datasmooth['A_head']['y']
for i,value in enumerate (headx, heady):
    if i < len(headx)-2:
        x1 = headx.iloc[i]
        x2 = headx.iloc[i + 1]
        x3 = headx.iloc[i + 2]
        y1 = heady.iloc[i]
        y2 = heady.iloc[i + 1]
        y3 = heady.iloc[i + 2]
        if (x3 - x2 > y3 - y2) & (x2 - x1 > y2 - y1):
            approach.append(1)
        else:
            approach.append(0)
approach.append(np.nan)
approach.append(np.nan)
datasmooth['approach'] = approach

nf = datasmooth['approach'] + datasmooth ['facing'] + datasmooth['OPopen']

A = nf[nf == 3].sum()
B = A/3
C = len(datasmooth['facing'])
Pertimeapproaching = B/C
Pertimeapproaching


# In[ ]:


approach = []
headx = datasmooth['A_head']['x']
heady = datasmooth['A_head']['y']
for i,value in enumerate (headx, heady):
    if i < len(headx)-2:
        x1 = headx.iloc[i]
        x2 = headx.iloc[i + 1]
        x3 = headx.iloc[i + 2]
        if x3 > x2 > x1:
            y1 = heady.iloc[i]
            y2 = heady.iloc[i+1]
            y3 = heady.iloc[i+2]
            diff = y3 - y1
            if -5 < diff < 5:
                approach.append(1)
        else:
            approach.append(0)
approach.append(np.nan)
approach.append(np.nan)
datasmooth['approach'] = approach

nf = datasmooth['approach'] + datasmooth ['facing'] + datasmooth['OPopen']

A = nf[nf == 3].sum()
B = A/3
C = len(datasmooth['facing'])
Pertimeapproaching = B/C
Pertimeapproaching


# In[ ]:


# datasmooth['union'] = datasmooth.near | datasmooth.facing | datasmooth.OPopen


# ##### # HM = distance((datasmooth['A_head']['x'], datasmooth['A_head']['y']),(datasmooth['OPMidpx'],datasmooth['OPMidpy']))
# # ME = distance((datasmooth['OPMidpx'],datasmooth['OPMidpy']), (datasmooth['eastpointx'], datasmooth['eastpointy']))
# # EH = distance ((datasmooth['A_head']['x'], datasmooth['A_head']['y']), (datasmooth['eastpointx'], datasmooth['eastpointy'])) 
# # C = np.sqrt(((datasmooth['head']['x']- datasmooth['eastpointx'])**2)+((datasmooth['head']['y']-datasmooth['eastpointy'])**2))
# # A = np.sqrt(((datasmooth['OPMidpx']- datasmooth['eastpointx'])**2)+((datasmooth['OPMidpy']-datasmooth['eastpointy'])**2))
# # B = np.sqrt(((datasmooth['head']['x']- datasmooth['OPMidpx'])**2)+((datasmooth['head']['y']-datasmooth['OPMidpy'])**2))
# # num = A**2 + B**2 - C**2
# # denom = 2*A*B
# # fraction = num/denom
# # cos = np.cos(fraction)
# # HeadDirect = np.degrees(cos)
# # HeadDirect
# datasmooth['HeadDirection'] = lawofcosines(TF,HT,FH)
# 
# plt.plot(datasmooth['HeadDirection'])
# plt.figure(figsize=(13,3))
# sns.kdeplot(datasmooth['HeadDirection'], bw=0.25)
# plt.title('Frequency of Angles')
# 
# 
# datasmooth
# # HeadDirect
# 

# # Attempts at Angular Velocity

# #### defining relevant functions

# ##### def ThreeDdistance (Xcords, Ycords, Zcords):
#     
#     ''' function that calculates 3D distance of x/y/z coordinates of a body through space
#     plug in the xcoords, ycoords, zcoords return the distance travelled by fish between frames'''
# 
#     distx = Xcords.diff() #converts distance to mm based on resolution/FOV
#     disty = Ycords.diff() #converts distance to mm based on resolution/FOV
#     distz = Zcords.diff() #converts distance to mm based on resolution/FOV
#     TotalDist = np.sqrt(distx**2 + disty**2 + distz**2)
#     
#     return TotalDist
# def ThreeDdistancebetweenpoints(pos_1, pos_2):
#     ''' measures 3D distance between two points, 
#     plug in x/y/z coordinates for two points as tubles (x0,y0,z0),(x1,y1,z1)
#     return the distance '''
#     x0,y0,z0 = pos_1
#     x1,y1,z1 = pos_2    
#     dist = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
#     
#     return dist
# 
# def cosineforcircle(R,H):
#     ''' Measures Theta of body angle change between current point, the next point, and an anchor point
#     set on the same circle, returns the theta in degrees'''
#     R1 = (R**2)*2
#     H1 = H**2
#     fraction = (R1-H1)/R1
#     theta2 = np.arccos(fraction)
#     degrees = np.degrees(theta2)
#     return degrees

# ##### Define the Radius of Circle, Distance between Next point and Current Anchor, and Find Theta of Direction

# In[ ]:


# # define R (radius)
# datasmooth['R'] = ThreeDdistance(datasmooth['head']['x'],datasmooth['head']['y'],datasmooth['Right Oper']['x'])

# # use radius to create adjusted anchor point for each time stamp, based on speed of the fish
# datasmooth['Anchorx'] = datasmooth['head']['x'] + datasmooth['R']
# datasmooth['Anchorx'] = datasmooth['Anchorx'].shift(-1)

# # use same y,z coordinates of current time point to put anchor on circle
# datasmooth['Anchory'] = datasmooth['head']['y']

# datasmooth['Anchorz'] = datasmooth['Right Oper'] ['x']


# # measuring theta based on length of Radius and distance between next point and anchor
# datasmooth['H'] = ThreeDdistancebetweenpoints((datasmooth['head']['x'],datasmooth['head']['y'],datasmooth['Right Oper']['x']), (datasmooth['Anchorx'], datasmooth['Anchory'], datasmooth['Anchorz']) )
# datasmooth['theta'] = cosineforcircle(datasmooth['R'],datasmooth['H'])

# datasmooth


# ##### Point Playback

# In[215]:


duration = 15
fps = 40
fig, ax = plt.subplots()
    
def make_frame(time):
    timeint = int(time*fps)+ 6000
    ax.clear()
    x = datasmooth['A_head']['x'][timeint]
    y = datasmooth['A_head']['y'][timeint]
    ax.plot(x,y,'o')
    x = datasmooth['E_leftoperculum']['x'][timeint]
    y = datasmooth['E_leftoperculum']['y'][timeint]
    ax.plot(x,y,'o')
    x = datasmooth['B_rightoperculum']['x'][timeint]
    y = datasmooth['B_rightoperculum']['y'][timeint]
    ax.plot(x,y,'o')
    ax.set_ylim([0,500])
    ax.set_xlim([0,500])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)
animation.write_gif('matplotlib2.gif', fps = 40)
    


# In[ ]:





# In[ ]:





# ##### def TwoDdistance (Xcords, Ycords):
#     
#     ''' function that calculates 3D distance of x/y/z coordinates of a body through space
#     plug in the xcoords, ycoords, zcoords return the distance travelled by fish between frames'''
# 
#     distx = Xcords.diff() #converts distance to mm based on resolution/FOV
#     disty = Ycords.diff() #converts distance to mm based on resolution/FOV
#      #converts distance to mm based on resolution/FOV
#     TotalDist = np.sqrt(distx**2 + disty**2)
#     
#     return TotalDist
# def TwoDdistancebetweenpoints(pos_1, pos_2):
#     ''' measures 3D distance between two points, 
#     plug in x/y/z coordinates for two points as tuples (x0,y0,z0),(x1,y1,z1)
#     return the distance '''
#     x0,y0 = pos_1
#     x1,y1 = pos_2    
#     dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#     
#     return dist
# 
# def calcH(pos_0, pos_1):
#     ''' measures distance between pos_1 in time(x+1) and pos_2 in time(x), plug in two points as tuples
#     returns an array of distances between two points in different columns and neighboring rows'''
#     
#     x0,y0 = pos_0
#     x1,y1 = pos_1
#     x1 = x1.shift(+1)
#     y1 = y1.shift(+1)
#     dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
#     x1 = x1.shift(-1)
#     y1 = y1.shift(-1)
#     dist = dist.shift(-1)
#     
#     return dist
#     
# def cosineforcircle(R,H):
#     ''' Measures Theta of body angle change between current point, the next point, and an anchor point
#     set on the same circle, returns the theta in degrees'''
#     R1 = (R**2)*2
#     H1 = H**2
#     fraction = (R1-H1)/R1
#     theta2 = np.arccos(fraction)
#     degrees = np.degrees(theta2)
#     return degrees

# In[ ]:


# # define R (radius)
# datasmooth['R2D'] = TwoDdistance(datasmooth['head']['x'],datasmooth['head']['y'])
# datasmooth['R2D'] = datasmooth['R2D'].shift(-1)
# # use radius to create adjusted anchor point for each time stamp, based on speed of the fish
# datasmooth['Anchorx2D'] = datasmooth['head']['x'] + datasmooth['R2D']

# # use same y,z coordinates of current time point to put anchor on circle
# datasmooth['Anchory2D'] = datasmooth['head']['y']

# # measuring theta based on length of Radius and distance between next point and anchor
# datasmooth['H2D'] = calcH((datasmooth['head']['x'],datasmooth['head']['y']), (datasmooth['Anchorx2D'], datasmooth['Anchory2D']) )
# datasmooth['theta'] = cosineforcircle(datasmooth['R2D'],datasmooth['H2D'])

# datasmooth['Angular Velocity'] = datasmooth['theta'].diff()
# datasmooth


# ##### Corrected Angular Velocity

# #### tryx = datasmooth['head']['x']
# tryy = datasmooth['head']['y']
# tryx = tryx.diff()
# tryy = tryy.diff()
# tryy = np.array(tryy.tolist())
# tryx = np.array(tryx.tolist())
# F = []
# for i,value in enumerate (tryx):
#     if np.isnan(value)== True:
#         F.append(np.nan)
#     else:
#         x = value
#         y = tryy[i]
#         A = np.degrees(math.atan2(y,x))
#         if A < 0:
#             A = A+360
#         F.append(A)
#         
#                 

# ##### D = [x+360 for x in F]
# datasmooth['theta2'] = D
# datasmooth['theta2'] = datasmooth['theta2'].shift(-1)
# 
# 
# # datasmooth['Angular Velocity'] = datasmooth['theta2'].diff()
# # datasmooth
# datasmooth.tail()

# ##### d = np.radians(F)
# # datasmooth['F'] = d
# # datasmooth['F'] = np.unwrap(datasmooth['F'])
# datasmooth['theta2'] = np.radians(datasmooth['theta2'])
# datasmooth['angular velocity'] = np.unwrap(datasmooth['theta2'])
# # datasmooth['wrap'] = np.unwrap(np.radians(d))
# 
# # datasmooth['Angular Velocity'] = datasmooth['wrap'].diff()
# datasmooth

# ##### Ang = abs(datasmooth['Angular Velocity'])
# plt.plot(Ang[])

# ##### fig, ax = plt.subplots()
# plt.plot(datasmooth['2DVelocity (mm/s)'], color = 'r', alpha = .7)
# ax2 = ax.twinx()
# plt.plot(datasmooth['Angular Velocity'], alpha = .7)

# ##### plt.figure(figsize=(13,3))
# sns.kdeplot(datasmooth['Angular Velocity'], bw=0.25)
# plt.title('Frequency of Angular Velocity')

# ##### datasmooth['AngVel Thresh'] = datasmooth['Angular Velocity'].apply(lambda x: 1 if x > 180 else 0)

# ##### plt.plot(datasmooth['AngVel Thresh'][:500])

# ##### datasmooth['Vel Thresh'] = datasmooth['2DVelocity (mm/s)'].apply(lambda x: 1 if x > 350 else 0)

# ###### fig, ax = plt.subplots()
# plt.plot(datasmooth['Vel Thresh'], color = 'r', alpha = .7)
# ax2 = ax.twinx()
# plt.plot(datasmooth['AngVel Thresh'], alpha = .7)

# In[ ]:





# In[ ]:





# In[ ]:




