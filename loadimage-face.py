import random
import os
import sys
import math
import numpy as np
import copy

from PIL import Image



trainPath = "D://python2.7.6//MachineLearning//pca-CNN//trainface-yr1"
datafile="D://python2.7.6//MachineLearning//pca-CNN//dataVec.txt"
labelfile="D://python2.7.6//MachineLearning//pca-CNN//dataLabel.txt" 
 
     

global classDic;classDic={}
global allLabel;allLabel=[]
global classList;classList=[]
global dataMat
 
dim=32 
dd=dim*dim
######################

def loadData():
    global dim,dd
    global dataMat,classDic,allLabel
    ###################classlist
    m=0
    for filename in os.listdir(trainPath):
        if filename not in classDic:
            classList.append(filename)
            classDic[filename]=m;m+=1
    ########## all label ,numcase
    for label in classList: # according to fixed order /sequence match label with image
        for docname in os.listdir(trainPath+'//'+label):
            allLabel.append(classDic[label])
    numcase=len(allLabel);
    dataMat=np.mat(np.zeros((numcase,dd)))  #n x 100x100
    ###########
    i=0
    for label in classList:
        for docname in os.listdir(trainPath+'//'+label):
            im=Image.open(trainPath+'//'+label+'//'+docname).convert('L')
            rsz=im.resize((dim,dim))
            imArr=np.array(rsz,'f')
            imArr=(255.0-imArr)/255.0# (0,255)->(0,1)
            imVec=imArr.flatten();
            dataMat[i,:]=imVec
            i+=1
    numcase,dd=np.shape(dataMat)
    print 'total image %d with dim %d'%(numcase,dd) #len(array)!=len(list)
    ##############
     
    outPutfile=open(datafile,'w')
    n,d=np.shape(dataMat)
    for i in range(n):
        for j in range(d):
            outPutfile.write(str(dataMat[i,j]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    ######
    outPutfile=open(labelfile,'w')
     
    for label in allLabel:
        outPutfile.write(str(label))
        outPutfile.write(' ')
         
    outPutfile.close()   
            
    
#############
loadData()





 
 

 
    
         
    
    
    







    
    
