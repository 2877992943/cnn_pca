import random
import os
import sys
import math
import numpy as np
import copy




dataName="D://python2.7.6//MachineLearning//pca-CNN//dataVec.txt" #n x 100x100 matric
labelfile= "D://python2.7.6//MachineLearning//pca-CNN//dataLabel.txt" #1 x n matric
 
outPath="D://python2.7.6//MachineLearning//pca-CNN//preprocessed.txt"     


dim=32 #100x100 memory error 


######################

def loadData():
    global dataMat, classList,labelList,dataList
    classDic={};labelList=[];dataList=[]
    ########## all label  list
    content=open(labelfile,'r')
    line=content.readline().strip(' ')
    line=line.split(' ')
    for label in line:
        labelList.append(int(label))
    print 'labellist',len(labelList)
    
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
     
    #####nxdd->ddxn
    dataMat=np.mat(dataList).T #10000xn datamat
     
    ########
     
     


def removeMean(dataMat1): #make mean of obs =0
     
    dd,n=np.shape(dataMat1)
    meancolum=dataMat1.mean(0)#n x 1 vec ,each patch have a unique mean
    dataMat1=dataMat1-np.tile(meancolum,(dd,1)) #10000xn
    return dataMat1

def calcCov(dataMat1):#input dataMat (256xn) return covMat(ddxdd)
    dd,n=np.shape(dataMat1)
     
    covMat=np.mat(np.zeros((dd,dd)))
    covMat=dataMat1*dataMat1.T/float(n)
    #print covMat
    return covMat

def calcPCA(covMat): #return xrot  not reduce dimention
    global dataMat
    #print 'cov', covMat
    eigval,eigmat=np.linalg.eig(covMat);#print 'eig',eigval,eigmat #3+1j not real number
    #umat,sigma,vmat=np.linalg.svd(covMat);#print'svd',umat,sigma
    #eigvec=umat;eigval=sigma
     
    eigvalInd=np.argsort(eigval) #return index of eigvalues
    eigvalInd=eigvalInd[::-1] #array # also arr[-1:-(n+1):-1]  [start:end:step] end is not included
    eigmat=eigmat[:,eigvalInd] #verticle vec is eigvec
     
    xrotMat=np.mat(np.zeros((np.shape((dataMat)))))#10000xn
    xrotMat=eigmat.T*dataMat# 10000x10000 x 10000xn
    return xrotMat,eigval,eigmat
    
    
def visualCovmat(covMat):# 256x256 image
    m,n=np.shape(covMat)
    #print covMat
    
    
    from PIL import Image
    ####change into uint8 [0,255] array
    covMat=covMat*100
    imMat=(covMat-covMat.min())/float(covMat.max()-covMat.min())#[0,1]
    imMat=imMat*255#[0,255]
    im=Image.fromarray(np.array(np.uint8(imMat)))
    ####
    import pylab as pb
    pb.imshow(im)
    pb.show()
    
def visualobsi(obs,d):
    x=obs.T#10000x1 vec->1x10000
     
    mat=vec2mat(x,d,d)#->100x100
    from PIL import Image
    ####change into uint8 [0,255] array
    imMat=(mat-mat.min())/float(mat.max()-mat.min())#[0,1]
    imMat=imMat*255#[0,255]
    im=Image.fromarray(np.array(np.uint8(imMat)))
    ####
    import pylab as pb
    pb.imshow(im)
    pb.show()
    
    
def whiten(xrotMat,eigval):#256xn
    dd,n=np.shape(xrotMat)
    eigval.sort()#itself get sorted increase
    eigvalnew=eigval[::-1]#  [::-1] decrease  #1x256 array
    fenmu=np.sqrt(np.mat(eigvalnew[:dd]))#dd=9 [0:9]means[0,1,,,8] 9 elements
    fenmu=np.tile(fenmu.T,(1,n))
    eps=0.0001
    xwhiMat=xrotMat/(fenmu+eps)
    return xwhiMat
    
    
def reduceDim(eigval,xrotMat): #1x256 array   256xn matric
    n,m=np.shape(np.mat(eigval))#1 256
    allInfo=np.mat(eigval)*np.mat(eigval).T
    allInfo=allInfo[0,0]
    threshold=0
    fenzi=0.0
    for k in range(m):
        threshold=k
        fenzi+=eigval[k]**2
        if fenzi/allInfo>0.999:break
    ####
    print '0.9at',threshold
    xrotRD=xrotMat[:threshold,:]
    return xrotRD 
    
        
def recover(xrot2,eigmat): #9xn matric   256x256matric
    dk,n=np.shape(xrot2)
    d1,d2=np.shape(eigmat)
    xOrigin=np.mat(np.zeros((d1,n)))#256xn
    xOrigin[:dk,:]=xrot2#[:9]means[0,1,2...8] 9elements 
    xOrigin=eigmat*xOrigin
    return xOrigin
    
    
def output(xmat):#1024xn matric
    outPutfile=open(outPath,'w')
    dd,n=np.shape(xmat)
    for i in range(n):
        for j in range(dd):
            outPutfile.write(str(xmat[j,i]))
            outPutfile.write(' ')
        outPutfile.write('\n')
    outPutfile.close()
    
    
    
    
###########################support
    
def vec2mat(vec,nhang,nlie):# vec->16x16
    m1,n1=np.shape(vec)
    szdiff=m1*n1-nhang*nlie
    if szdiff!=0:
        print 'this vec cannot transfer into mat'
         
    ############
    Mat=np.mat(np.zeros((nhang,nlie)))
    for m in range(nhang):
        for n in range(nlie):
            pos=m*nlie+n
            Mat[m,n]=vec[0,pos]
    return Mat    
    
    

 
#######################main
loadData()

######get covariance matric
#dataMat=np.mat([[1,2,3],[4,5,6],[7,8,9]])
dataMat=removeMean(dataMat)
#visualobsi(dataMat[:,0],dim) 
covMat=calcCov(dataMat)
print 'datamat covMat'
##########rotate  pca  :uncorrelated dim1 dim2... cov(d1,d2)=0 ,remove redundant information: reflect vecx into new feat space by multiply matric eigmat
xrotMat,eigval,eigmat=calcPCA(covMat)#not reduce dim
print 'xrotMat'
##check pca by see whether cov(dim1,dim2)  is 0 or uncorrelated
covMat1=calcCov(xrotMat)
print 'xrotMat covmat'
#visualCovmat(covMat1) #show xrot covariance
 
#############reduce dim  based on new base U
xrot2=reduceDim(eigval,xrotMat)#9xn replace 256xn
covMat2=calcCov(xrot2)
#visualCovmat(covMat2)
xOrigin=recover(xrot2,eigmat)
#visualobsi(xOrigin[:,0],dim)
##########whiten
xwhiMat=whiten(xrot2,eigval) #9xn
covMat3=calcCov(xwhiMat)
#visualCovmat(covMat3)
xOrigin=recover(xwhiMat,eigmat)
#visualobsi(xOrigin[:,0],dim)
#####output data
output(xOrigin)
  
     
    
 



'''
eigv,eigvec=np.linalg.eig(m) #eigvalue is array,eigmat is matric with verticle eigvec
eimat[:,eigvlInd]
'''
