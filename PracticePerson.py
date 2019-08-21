#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:46:36 2019

@author: Claire
#"""
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

h5_dir = '/Users/Claire/Desktop/SEAAnalyze'
h5_files = glob(os.path.join(h5_dir,'*.h5'))


##Functions

def mydistance(pos_1,pos_2):
    '''
    Takes two position tuples in the form of (x,y) coordinates and returns the distance between two points
    '''
    x0,y0 = pos_1
    x1,y1 = pos_2    
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    return dist

def coords(data):
    '''
    helper function for more readability/bookkeeping
    '''
    return (data['x'],data['y'])

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


def auto_scoring_TS1(data_auto,thresh_param0 = 65.45,thresh_param1 = 135.56):
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

def Percent_Trial(func, data_auto):
    BinaryVector = func(data_auto)
    OpOpen = (BinaryVector.sum())/len(BinaryVector)
    return OpOpen


def speed (data, fps = 40):
    
    ''' function that calculates velocity of x/y coordinates
    plug in the xcords, ycords, relevant dataframe, fps
    return the velocity as column in relevant dataframe'''
    poi = ['A_head']
    (Xcoords, Ycoords)= coords(data[poi[0]])
    distx = Xcoords.diff() 
    disty = Ycoords.diff()
    TotalDist = np.sqrt(distx**2 + disty**2)
    Speed = TotalDist / (1/fps) #converts to seconds
    AvgSpeed = Speed.mean()
    
    return AvgSpeed

sns.set_context('poster')
sns.set_style('white')


def polar_plot(orientation_array,title,dec = 3):
    if len(orientation_array) > 71000:
        # N is the number of bins
        N = 36
        bottom = 2

        # create theta for 24 hours
        theta = np.linspace(0.0,np.pi, N, endpoint=False)

        # make the histogram that bined on 24 hour
        radii, tick = np.histogram(orientation_array, bins = N)
        print(orientation_array)
        # width of each bin on the plot
        width = (2*np.pi) / N

        # make a polar plot
        plt.figure(figsize = (12, 8))

        ax = plt.subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        bars = ax.bar(theta, radii, width=width, bottom=bottom)

        # set the lable go clockwise and start from the top
        ax.set_theta_zero_location("E")
        # clockwise
        # ax.set_theta_direction(-1)

        # set the label
        ticks = [str(i*45) for i in range(8)]
        # ticks = ['0:00', '3:00', '6:00', '9:00', '12:00', '15:00', '18:00', '21:00']
        ax.set_xticklabels(ticks)
        ax.set_xticks(np.pi/180. * np.linspace(0,  180, 5, endpoint=True))

        ## Make the proportions nice: 

        acceptable_values = np.linspace(0,1,1*10**dec + 1)

        ## Figure out the maximum proportion:
        maxprop = max(radii/np.sum(radii))
        ## Convert to a pleasing round decimal
        maxaccept = acceptable_values[np.where(acceptable_values>maxprop)][0]
        ## Convert to seconds: 
        seconds = sum(radii)/40.
        max_secs = seconds*maxaccept 

        ax.set_yticklabels(np.linspace(0,max_secs,5).astype(int))



        ax.set_yticks(np.linspace(0,sum(radii)*maxaccept,5))
        label_position=ax.get_rlabel_position()
        ax.text(np.radians(label_position+300),ax.get_rmax()/2.,'sec',
                rotation=0,ha='center',va='center')

        ##TODO: Make into seconds
        plt.title(title)
        plt.savefig(title + '.png')
        plt.show()
    else:
        print('data is empty')
    
def Yaxis (data, title):
    poi = ['A_head']
    coord = ['y']
    
    plt.plot(data[poi[0],coord[0]])
    
    plt.savefig(title + '.png')
    plt.show()

def myplot(x, y, s, bins=500):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

def heatmap(data_auto, title):
    img, extent = myplot(data_auto['A_head']['x'], data_auto['A_head']['y'], 2)
    fig, ax = plt.subplots()
    ax.imshow(img, extent=extent, origin='lower', cmap=cm.rainbow, vmin=0, vmax=1)
    ax.set(xticks=[], yticks=[])
    plt.savefig(title + '.png')
    plt.show()
    
class SelectFile(object):

    #can I do that thing where I make h5_files = h5_files so it is always the default?
    def __init__(self, filename, barrier_up, behavior_start, barrier_down):
        self.filename = filename
        self.barrier_up = barrier_up
        self.behavior_start = behavior_start
        self.barrier_down = barrier_down
        self.data_auto= self.PreparedData()
        self.data_auto_filt = self.auto_scoring_tracefilter_new()
        self.pre, self.test, self.post = self.EpochSeg()
        self.preOp, self.testOp, self.postOp = self.Operculum()
        self.preSpeed, self.testSpeed, self.postSpeed = self.Speed()
        
        print('updated')
        
    def PreparedData(self):
        with pd.HDFStore(self.filename,'r') as help2:
            data_auto = help2.get('df_with_missing')
            data_auto.columns= data_auto.columns.droplevel()
        return(data_auto)
      # I can get it returned, but how do I save it as a new variable? (example of data_auto)
      # is this smart to have one function reference the out put of another? better way to do it?
      #Can I make each measure (operculum, speed etc. a different function but then have one master function
      #that runs them all and stores into a dataframe? that saves as an excel?)
      # Is there a way to say, now for all "Trials", run this function?
      ## List comprehension: [PracticePerson.SelectFile(h5_files[i],barrier_up[i], barrier_down[i]) for i in range(len(h5_files))
      ## %autoreload 1
      ## %aimport Practice Person
  
    
    def auto_scoring_tracefilter_new(self,p0=20,p1=250,p2=15,p3=70,p4=200):
        
        
        boi = ['A_head','B_rightoperculum', 'C_tailbase', 'D_tailtip','E_leftoperculum']
        self.data_dists = pd.DataFrame()
        self.data_dists['bodylength'] = mydistance(coords(self.data_auto[boi[0]]),coords(self.data_auto[boi[3]]))
        self.data_dists['Operwidth'] = mydistance(coords(self.data_auto[boi[4]]),coords(self.data_auto[boi[1]]))
        self.data_dists['HeadROperwidth'] = mydistance(coords(self.data_auto[boi[0]]),coords(self.data_auto[boi[1]]))
        self.data_dists['HeadLOperwidth'] = mydistance(coords(self.data_auto[boi[0]]),coords(self.data_auto[boi[4]]))
        self.data_dists['TailtipROperwidth'] = mydistance(coords(self.data_auto[boi[3]]),coords(self.data_auto[boi[1]]))
        self.data_dists['TailtipLOperwidth'] = mydistance(coords(self.data_auto[boi[3]]),coords(self.data_auto[boi[4]]))
        self.data_dists['TailbaseLOperwidth'] = mydistance(coords(self.data_auto[boi[2]]),coords(self.data_auto[boi[4]]))
        self.data_dists['TailbaseROperwidth'] = mydistance(coords(self.data_auto[boi[2]]),coords(self.data_auto[boi[1]]))
    
        data_auto_filt = self.data_auto.copy()
        
        for b in boi:
            for j in ['x','y']:
                xdifference = abs(data_auto_filt[b][j].diff())
                xdiff_check = xdifference > p0     
        #         print (xdiff_check.loc[xdiff_check == True])
                data_auto_filt[xdiff_check] = np.nan
        #         print (data_auto_filt.loc[np.isnan(data_auto_filt['A_head']['x'])])
    
                bodylength_check = self.data_dists['bodylength'] > p1
                data_auto_filt[bodylength_check] = np.nan
    
                origin_check = data_auto_filt[b][j] < p2
                data_auto_filt[origin_check] = np.nan
    
                Operwidth_check = self.data_dists['Operwidth'] > p3
                data_auto_filt[Operwidth_check] = np.nan
    
                HeadROperwidth_check = self.data_dists['HeadROperwidth'] > p3
                data_auto_filt[HeadROperwidth_check] = np.nan
    
                HeadLOperwidth_check = self.data_dists['HeadLOperwidth'] > p3
                data_auto_filt[HeadLOperwidth_check] = np.nan
    
                TTL_check = self.data_dists['TailtipLOperwidth'] > p4
                data_auto_filt[TTL_check] = np.nan
    
                TTR_check = self.data_dists['TailtipROperwidth'] > p4
                data_auto_filt[TTR_check] = np.nan
                
        return(data_auto_filt) 
    
    def EpochSeg (self):
        pre = self.data_auto[:self.barrier_up]
        test = self.data_auto[self.behavior_start:self.barrier_down]
        post = self.data_auto[self.barrier_down:]
        
        return (pre, test, post)
    
    def Operculum (self):
        preOp = Percent_Trial(auto_scoring_TS1,self.pre)
        testOp = Percent_Trial(auto_scoring_TS1,self.test)
        postOp = Percent_Trial(auto_scoring_TS1,self.post)
        
        ##save the dataframe as something else
        return(preOp, testOp, postOp)
        
    def Speed (self):
        preSpeed = speed(self.pre)
        testSpeed = speed(self.test)
        postSpeed = speed(self.post)
        
        return (preSpeed, testSpeed, postSpeed)
    
    def Orientation (self,savename):
        
        polar_plot(self.pre.values, savename+ '/' + 'prepolarplot' + str([self.filename.split('/')[5]]))
        polar_plot(self.test.values, savename+ '/' + 'testpolarplot' + str([self.filename.split('/')[5]]))
        polar_plot(self.post.values, savename+ '/' + 'postpolarplot' + str([self.filename.split('/')[5]]))
        
    def Yaxis (self, savename):
        
        Yaxis(self.pre, savename + '/' + 'preYaxis' + str([self.filename.split('/')[5]]))
        Yaxis(self.test, savename + '/' + 'testYaxis' + str([self.filename.split('/')[5]]))
        Yaxis(self.post, savename + '/' + 'postYaxis' + str([self.filename.split('/')[5]]))
        
    def HeatMapLocation (self, savename):
        heatmap(self.pre, savename + '/' + 'preYaxis' + str([self.filename.split('/')[5]]))
        heatmap(self.test, savename + '/' + 'testYaxis' + str([self.filename.split('/')[5]]))
        heatmap(self.post, savename + '/' + 'postYaxis' + str([self.filename.split('/')[5]]))
    
        
class SelectFolder(object):
    
    ''' 
    Takes folder name parameter, format: "folder/folder/folder" **leave off last 
    slash** Will take folder, iterate through .h5 files, create new analysis folder 
    where operculum, speed, position, orientation information is saved
    '''
    def __init__(self, folder):
        self.folder = folder
        self.h5_files = self.getfiles()
        self.barrier_up, self.behavior_start, self.barrier_down = self.getbarrier()
        self.SelectFileList = self.getanalysis()
        self.makenewfolder()
        
        
    def getfiles (self):
        h5_files = glob(os.path.join(self.folder,'*.h5'))
        return h5_files
    
    def getbarrier (self):
        excel_files = glob(os.path.join(h5_dir,'*.xlsx'))
        file_handle = excel_files[0]
        data_manual = pd.read_excel(file_handle)
        barrier_up = data_manual['barrier_up']
        behavior_start = data_manual['barrier_up']
        barrier_down = data_manual['barrier_down']
        
        return (barrier_up, behavior_start, barrier_down)
        
    def getanalysis (self):
        SelectFileList = [] 
        
        if len(self.barrier_up) == len(self.barrier_down):
            if len(self.barrier_up) == len(self.h5_files):
                counter = 0
                for i in self.h5_files:
                    Trial = SelectFile(i, self.barrier_up[counter], self.behavior_start[counter], self.barrier_down[counter] )
                    SelectFileList.append(Trial)
                    counter = counter + 1
                return SelectFileList 
    
    def makenewfolder (self):
        self.analysisfolder = self.folder + "/" + "Analysis"
        if os.path.exists(self.analysisfolder):
            pass
        else:
            os.mkdir(self.analysisfolder)
            
        print("I just created" + self.analysisfolder)
    
    def OrientationStore (self):
        for i in self.SelectFileList:    
            i.Orientation(self.analysisfolder)
            #i.Operculum(self.analysisfolder)
    
    def YaxisStore (self):
        for i in self.SelectFileList:
            i.Yaxis(self.analysisfolder)
    
    def HeatMapStore (self):
        for i in self.SelectFileList:
            i.HeatMapLocation(self.analysisfolder)
            
    def MakeDataFrame (self):
        self.Results = pd.DataFrame()
        preOplist = []
        testOplist = []
        postOplist = []
        preSpeedlist = []
        testSpeedlist =[]
        postSpeedlist = []
        filelist = []
        
        for i in self.SelectFileList:
            preOplist.append(i.preOp)
            testOplist.append(i.testOp)
            postOplist.append(i.postOp)
            preSpeedlist.append(i.preSpeed)
            testSpeedlist.append(i.testSpeed)
            postSpeedlist.append(i.postSpeed)
            filelist.append(i.filename)
            
        self.Results['preOp'] = preOplist
        self.Results['testOp'] = testOplist
        self.Results['postOp'] = postOplist
        self.Results['preSpeed'] = preSpeedlist
        self.Results['testSpeed'] = testSpeedlist
        self.Results['postSpeed'] = postSpeedlist
        self.Results['Identity'] = filelist
        
        self.Results.to_excel(self.analysisfolder + '/' + "DataFrame.xlsx")
        return (self.Results)
        
# store heat maps, make data frame, implement different barriers
            
            
    #filter data, can work on smoothing later on?
    
    
#    def Filtering(self, barrier_up, barrier_down):
#        
#class Extractor(SelectFile):
#    
#    def __init__(self, data_auto, sex, barrier_up, barrier_down):
#        self.filename = filename
#        self.sex = sex
#        self.barrier_up = barrier_up
#        self.barrier_down = barrier_down
     
#class Person:
#  def __init__(self, name, age):
#    self.name = name
#    self.age = age


#install as a pip package, pip install "name". Put on github, pull. Make a pip package
#can put the pip onto github
