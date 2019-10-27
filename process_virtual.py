"""
Code to process points extracted from blender giving positions of a certain rendered vertex. 
Inputs: 
    1: the path to the file to be used as input. Expects a blender output file.  
"""



import csv
import sys
import numpy as np
import re
from tqdm import tqdm

filename = sys.argv[1]
#outname = sys.argv[2]

container = []
with open(filename,"r") as f:
    counter = 0
    for line in f:
        container.append(line)
        counter +=1

data = container[0].split("><Vector ")

arrayinit = []
for d in tqdm(data):
    exp = re.findall("\(([^\)]+)\)",d)[0].split(',')
    locallist = []
    for f in exp:
        locallist.append(float(f))
    arrayinit.append(locallist)
        

#dataparsed = [[float(f) for f in re.findall("\(([^\)]+)\)",d)[0].split(',')] for d in data]

stack = np.stack(arrayinit)
print(stack.shape)
print(stack)
print(np.where(np.isnan(stack))[0].shape)
