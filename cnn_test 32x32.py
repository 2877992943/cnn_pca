import random
import os
import sys
import math
import numpy as np



 
dataName="D://python2.7.6//MachineLearning//pca-CNN//dataVec.txt"
labelfile= "D://python2.7.6//MachineLearning//pca-CNN//dataLabel.txt"
 

outPath="D://python2.7.6//MachineLearning//pca-CNN//para"
 
outfile4 = "C.txt"
outfile5 = "W.txt"

outfile6 = "B.txt"
outfile7 = "BB.txt"
 

     

global classDic;classDic={}
global dataList
global labelList
global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc


xdim=32;xdd=xdim**2
fltdim=9;fltdd=81
convdim=xdim-fltdim+1;convdd=convdim**2
pooldim=2;pooldd=pooldim**2
numflt=50

outdim=convdim/pooldim;outdd=outdim**2


nhh=outdd*numflt
numc=10
 
 
 
 
global dataMat,yMat,outputMat
######################

def loadData():
    global dataMat,yMat,outputMat 
    global dataList
    global labelList
    global numc,numflt,dimflt,nhh,nh,xdim
    dataList=[];labelList=[]
    ########## all label  list
    content=open(labelfile,'r')
    line=content.readline().strip(' ')
    line=line.split(' ')
    for label in line:
        labelList.append(int(label))
    print '1',len(labelList)
    
    ##########
    obs=[]
    content=open(dataName,'r')
    line=content.readline().strip('\n').strip(' ')
    line=line.split(' ')
    #print line,len(line)
    while len(line)>1:
        obs=[float(n) for n in line if len(n)>1]
        #print 'o',obs,len(obs)
        
        line=content.readline().strip('\n').strip(' ');line=line.split(' ')
         
        dataList.append(obs);#print 'datalist',len(dataList)
    ##########
    print '%d obs loaded'%len(dataList),len(set(labelList)),'kinds of labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
    global numc
    numc=len(set(labelList))
    #####
    dataMat=np.mat(dataList)
     
    ########
    num,dd=np.shape(dataMat)
     
    yMat=np.zeros((num,numc))
    for n in range(num):
        truey=labelList[n]
        yMat[n,truey]=1.0
    ######
    #######
def loadPara():
    global bbmat,bmat 
    global Cmat,Wmat
    global dataMat,yMat,outputMat
    global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    num,dim=np.shape(dataMat)
    print 'num,xdim',num,xdim
    ###############
    content=open(outPath+'/'+outfile4,'r')  #cmat
    line=content.readline().strip('\n').strip(' ')
    Clist=[] 
    while len(line)!=0:
        obs=[]
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                obs.append(float(n))
                 
        line=content.readline().strip('\n').strip(' ')
        Clist.append(obs)
    ####
    Cmat=np.mat(Clist)
    numc,nhh=np.shape(np.mat(Clist));print 'numc,nhh',numc,nhh
    #####################
    content=open(outPath+'/'+outfile5,'r')  #wmat
    line=content.readline().strip('\n').strip(' ')
    Wlist=[] 
    while len(line)!=0:
        obs=[]
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                obs.append(float(n))
                 
        line=content.readline().strip('\n').strip(' ')
        Wlist.append(obs)
    ####
    Wmat=np.mat(Wlist)
    numflt,fltdd=np.shape(np.mat(Wlist));
    print 'numflt,fltdd',numflt,fltdd
    ##########
     
    ################
    content=open(outPath+'/'+outfile6,'r')  #matb
    line=content.readline().strip('\n').strip(' ')
    Blist=[] 
    while len(line)!=0:
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                Blist.append(float(n))
                #Nmat[n,i]=float(line[i])
        line=content.readline().strip('\n').strip(' ')  #len(' ')== 1
         
    ####
    bmat=np.mat(Blist)
    print np.shape(bmat);
    #######################
    content=open(outPath+'/'+outfile7,'r')  #mat2b
    line=content.readline().strip('\n').strip(' ')
    B2list=[] 
    while len(line)!=0:
         
        line=line.split(' ')
        #print len(line),line[1023],'len'  #line[1024]==' ' length is 1025
        for n in line:
            #print '1',line[i]
            if len(n)>=1:
                B2list.append(float(n))
                #Nmat[n,i]=float(line[i])
        line=content.readline().strip('\n').strip(' ')  #len(' ')== 1
         
    ####
    bbmat=np.mat(B2list)
    print np.shape(bbmat);
 
        
        

def initialH():
    global dataMat,yMat
    global hMat,hhMat,outputMat
    global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    num,dim=np.shape(dataMat)
    hMat=np.mat(np.zeros((numflt,convdd)))#5,64
    outputMat=np.mat(np.zeros((num,numc))) #10
    hhMat=np.mat(np.zeros((numflt,outdd)))#5 16
      

def forward(x): #xi index not xvector
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    xvec=dataMat[x,:]
    ######1x256->16x16->64x81   9x9filter
    x16=vec2mat(xvec,xdim,xdim)###############???
    #print 'x16',np.shape(x16)
    #### ->64x81    (16-9+1)x(16-9+1) 8x8
    x64=np.mat(np.zeros((convdd,fltdd)))##64 pieces of dim81 patch
     
    i=0
    for hang in range(convdim):
        for lie in range(convdim):
            patch=x16[hang:hang+fltdim,lie:lie+fltdim]  #[0:9]==0,1,2,,,8 no 9
            #print 'patch',np.shape(patch)
            pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
            x64[i,:]=pVec
            i+=1

    #####conv
    for patch in range(convdd):
        for kernel in range(numflt):
            con=Wmat[kernel,:]*x64[patch,:].T#1x81  x  81x1
            con=con[0,0]+bmat[0,kernel]
            con=1.0/(1.0+math.exp((-1.0)*con))
            hMat[kernel,patch]=con
    #####pool
    for k in range(numflt): #each kernel
        ####1x64->8x8 featmap  , 4x4 poolmap
        feaMap=vec2mat(hMat[k,:],convdim,convdim) 
        ####pool with 2x2 window mean pooling
        poolMap=np.mat(np.zeros((outdim,outdim)))
        for hang in range(outdim):
            for lie in range(outdim):
                patch=feaMap[hang*pooldim:hang*pooldim+pooldim,lie*pooldim:lie*pooldim+pooldim]
                v=patch.flatten().mean()
                poolMap[hang,lie]=v
        #####4x4->1x16 poolmap
        hhMat[k,:]=poolMap.flatten()
    #######full connect
    hhvec=hhMat.flatten()#5x16 -> 1x80
    fvec=hhvec*Cmat.T+bbmat#1x80  x  80x10==1x10
    outputMat[x,:]=softmax(fvec)


    
def predict():
    global dataMat,yMat,outputMat
    global labelList
    global pooldim,pooldd,outdim,outdd,convdim,convdd,numflt,fltdim,fltdd,xdim,xdd,nhh,numc
    num,xdim=np.shape(dataMat)
    right=0.0
    for n in range(num):
        truey=labelList[n]
        maxP=-10;maxL=0
        for w in range(numc):
            if maxP==-10 or maxP<outputMat[n,w]:
                maxP=outputMat[n,w]  #update leizhu
                maxL=w
        ######final winner
        if truey==maxL:right+=1.0
        #print truey,maxL
    #####
    print 'accuracy',right/float(num)
##############################
def vec2mat(vec,nhang,nlie): #input vec 1x16 ouput matric 4x4
    if nhang!=nlie:print 'num hang must = num lie'
    n=nhang#for example :1x16->4x4
    Mat=np.mat(np.zeros((n,n)))
    for hang in range(n):
        for lie in range(n):
            pos=n*hang+lie
            Mat[hang,lie]=vec[0,pos]
    return Mat
 
def softmax(outputMat): #1x10 vec
    vec=np.exp(outputMat)  #1x10  #wh+b
    ss=vec.sum(1);ss=ss[0,0]
    outputMat=vec/(ss+0.000001)
    return outputMat
    
def normalize(vec,opt):
    if opt=='prob': #in order to sum prob=1
        ss=vec.sum(1)[0,0]
        vec=vec/(ss+0.000001)
    if opt=='vector': #in order to mode or length ||vec||=1
        mode=vec*vec.T
        mode=math.sqrt(mode[0,0])
        vec=vec/(mode+0.000001)
    if opt not in ['vector','prob']:
        print 'only vector or prob'
    return vec 
 

###################main
loadData()
loadPara()

num=np.shape(dataMat)[0] 
#### initial
initialH()

####get output f
for n in range(num):
    forward(n)

predict()
 
 

 



    
    
