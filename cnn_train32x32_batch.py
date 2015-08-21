import random
import os
import sys
import math
import numpy as np
import copy




dataName="D://python2.7.6//MachineLearning//pca-CNN//dataVec.txt" #n x 32x32 matric
labelfile= "D://python2.7.6//MachineLearning//pca-CNN//dataLabel.txt" #1 x n matric
 
outPath="D://python2.7.6//MachineLearning//pca-CNN//para"     

global classDic,labelList,dataList #11 2000
 
global epoch;epoch=3
global alpha;alpha=0.2
xd=32
xdd=xd*xd
nfilter=50
filterd=9
filterdd=81
convd=xd-filterd+1
convdd=convd**2
pwind=2
poold=convd/pwind
pooldd=poold**2

nh=convd**2
nhh=poold**2
numc=10

nbatch=5


######################

def loadData():
    global dataMat,yMat,classList,labelList,dataList
    classDic={};labelList=[];dataList=[]
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
        for n in line:
            start=n.find('(')+1
            end=n.find('+')
            obs.append(float(n[start:end]))
        #obs=[float(n[1:-1]) for n in line if len(n)>1]
        #print 'obs',obs,len(obs)
        line=content.readline().strip('\n').strip(' ');line=line.split(' ')
        dataList.append(obs);#print 'datalist',len(dataList)
        obs=[]
    ##########
    print '%d obs loaded'%len(dataList),len(set(labelList)),'kinds of labels',len(dataList[0]),'dim'
    #print labelList,classDic
    ####
    global numc#numc get changed here
    numc=len(set(labelList))
    #####
    dataMat=np.mat(dataList)
     
    ########
    num,dd=np.shape(dataMat)
     
    yMat=np.zeros((num,numc))
    for n in range(num):
        truey=labelList[n]
        yMat[n,truey]=1.0


def initialH():
    global dataMat,yMat
    global hMat,hhMat,outputMat
    
    num,dd=np.shape(dataMat)
    hMat=np.mat(np.zeros((nfilter,nh)))#nfilter,nh
    outputMat=np.mat(np.zeros((num,numc))) #nclass
    hhMat=np.mat(np.zeros((nfilter,nhh)))#nhh
    
    
     

def initialPara():
    global Cmat,Wmat,bmat,bbmat #initial from random eps
    num,dd=np.shape(dataMat)
      
    Cmat=np.mat(np.zeros((numc,nhh*nfilter)))
    Wmat=np.mat(np.zeros((nfilter,filterdd)))
    bmat=np.mat(np.zeros((1,nfilter)))
    bbmat=np.mat(np.zeros((1,numc)))
    for i in range(numc):
        for j in range(nhh*nfilter):
            Cmat[i,j]=random.uniform(0,0.1)
    for i in range(nfilter):
        for j in range(filterdd):
            Wmat[i,j]=random.uniform(0,0.1)
    
    #######
    for j in range(nfilter):
        bmat[0,j]=random.uniform(0,0.1)
     
    for j in range(numc):
        bbmat[0,j]=random.uniform(0,0.1)
     
    

            
def initialErr():#transfer err sensitive
    global errW,errC,up1,up2
    global dataMat
     
    n,d=np.shape(dataMat)
    errW=np.mat(np.zeros((1,nh)))
    errC=np.mat(np.zeros((1,numc)))
    up1=np.mat(np.zeros((poold,poold)))
    up2=np.mat(np.zeros((convd,convd)))
     
    
def initialGrad():
      
    global gradc,gradw,gradb,gradbb
    gradc=np.mat(np.zeros((numc,nfilter*nhh)))
    gradw=np.mat(np.zeros((nfilter,filterdd)))
    gradb=np.mat(np.zeros((1,nfilter)))
    gradbb=np.mat(np.zeros((1,numc)))

def forward(x): #xi index not xvector
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    
    xvec=dataMat[x,:]
    ######1x256->16x16->64x81   9x9filter
    x16=vec2mat(xvec,xd,xd)
    #print x16
    #### ->64x81    (16-9+1)x(16-9+1) 8x8
    x64=np.mat(np.zeros((nh,filterdd)))##64 pieces of dim81 patch
    i=0
    for hang in range(convd):
        for lie in range(convd):
            patch=x16[hang:hang+filterd,lie:lie+filterd]  #[0:9]==0,1,2,,,8 no 9
            #print patch
            pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
            x64[i,:]=pVec
            i+=1

    #####conv
    for patch in range(nh):
        for kernel in range(nfilter):
            con=Wmat[kernel,:]*x64[patch,:].T#1x81  x  81x1
            con=con[0,0]+bmat[0,kernel]
            con=1.0/(1.0+math.exp((-1.0)*con))
            hMat[kernel,patch]=con
    #####pool
    for k in range(nfilter): #each kernel
        ####1x64->8x8 featmap  , 4x4 poolmap
        feaMap=vec2mat(hMat[k,:],convd,convd) 
        ####pool with 2x2 window mean pooling
        poolMap=np.mat(np.zeros((poold,poold)))
        for hang in range(poold):
            for lie in range(poold):
                patch=feaMap[hang*pwind:hang*pwind+pwind,lie*pwind:lie*pwind+pwind]#pool window 2x2
                v=patch.flatten().mean()
                poolMap[hang,lie]=v
        #####4x4->1x16 poolmap
        hhMat[k,:]=poolMap.flatten()
    #######full connect
    hhvec=hhMat.flatten()#5x16 -> 1x80
    fvec=hhvec*Cmat.T+bbmat#1x80  x  80x10==1x10
    outputMat[x,:]=softmax(fvec)
    ######
     
    return x16
                
def calcGrad(x,x16):#x index not vec
    global hMat,hhMat,outputMat,yMat
    global Cmat,Wmat,bmat,bbmat
    global dataMat
    global gradc,gradw,gradb,gradbb
    global errW,errC,up1,up2
    
    ####err c floor
    fy=outputMat[x,:]-yMat[x,:] #matric 1x10
    sgm=outputMat[x,:].A*(1.0-outputMat[x,:].A)
    errC=np.mat(fy.A*sgm)#matric 1x10
    ######grad c
    hhflat=hhMat.flatten()#1x80 matric
    gradc=gradc+errC.T*hhflat  #10x1  x  1x80==10x80
    gradbb=gradbb+copy.copy(errC)#1x10 ##cannot change  at the same time
    #####5 kernel
    for k in range(nfilter):
        ####calc up1
        vec=errC*Cmat[:,k*nhh:k*nhh+nhh] #1x10  x  10x16==1x16
        up1=vec2mat(vec,poold,poold) #1x16->4x4
        ######calc up2 :upsample: expand and divide 2x2 pooling windon
        for hang in range(poold):
            for lie in range(poold):
                m=up1[hang,lie]/float(pwind**2)
                mat2x2=np.mat(np.zeros((pwind,pwind)))+m#2x2 window filed with mean/4
                up2[pwind*hang:pwind*hang+pwind,pwind*lie:pwind*lie+pwind]=mat2x2
        #####8x8->1x64
        vecUp2=up2.flatten()#matric 1x64
        ####err for w floor
        sgm=hMat[k,:].A*(1.0-hMat[k,:].A)#1x64 array
        errW=np.mat(sgm*vecUp2.A)#1x64 matric
        ######calc w grad :conv2(x,errw,valid) 16x16 conv with 8x8==9x9
        ####x 16x16->(8x8)x81  conv with patch/filter 8x8
        x81=np.mat(np.zeros((filterdd,nh)))##81 pieces of dim64 patch8x8
        i=0
        for hang in range(filterd):
            for lie in range(filterd):
                patch=x16[hang:hang+convd,lie:lie+convd]  #[0:9]==0,1,2,,,8 no 9
                #print patch
                pVec=patch.flatten() ;#print np.shape(pVec)#matric 1x81
                x81[i,:]=pVec; 
                i+=1
         
        ###conv x with filter 8x8
        gradw[k,:]=gradw[k,:]+errW*x81.T#1x64  x  64x81==1x81
        ########
        gradb[0,k]=gradb[0,k]+errW.sum(1)[0,0]

def divideNormalizeGrad():
    global gradc,gradw,gradb,gradbb
    #############divide num of obs in batch
    gradc=gradc/float(nbatch)
    gradw=gradw/float(nbatch)
    gradb=gradb/float(nbatch)
    gradbb=gradbb/float(nbatch)
    ##############gradient normalize
    for k in range(numc):
        gradc[k,:]=normalize(gradc[k,:],'vector')
    for k in range(nfilter):
        gradw[k,:]=normalize(gradw[k,:],'vector')
    
def updatePara():
    global Cmat,Wmat,bmat,bbmat
    global gradc,gradw,gradb,gradbb
    Cmat=Cmat+alpha*(-1.0)*gradc
    Wmat=Wmat+alpha*(-1.0)*gradw
    bmat=bmat+alpha*(-1.0)*gradb
    bbmat=bbmat+(-1.0)*alpha*gradbb
    
def calcLoss():
    global Cmat,Wmat,bmat,bbmat
    global outputMat,yMat,dataMat # fk is calculated with old para
    num,dim=np.shape(dataMat)
    loss=0.0
    for n in range(num)[:100]:
        diff=outputMat[n,:]-yMat[n,:]#1x10 mat
        ss=diff*diff.T;ss=ss[0,0]
        loss+=ss
    #print 'least square loss',loss
    return loss
        
        
def gradCheckerC(x):#x index not xvec
    global gradc,gradw,gradb,gradbb
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global outputMat,yMat,dataMat
    
    eps=0.0001
    for c in range(numc):
        for dim in range(pooldd):#4x4
            ###################calc loss postive
            #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
            Cmat[c,dim]=Cmat[c,dim]+eps
            x16=forward(x)#to get h outputf
            ###calc loss  
            lossx1=outputMat[x,:]-yMat[x,:]
            lossx1=lossx1*lossx1.T
            lossx1=lossx1[0,0]*0.5
            ##################calc loss negative
            #####theta vec negative=theta vec - [1,0,0,0,..]
            Cmat[c,dim]=Cmat[c,dim]-2.0*eps
            x16=forward(x)#to get h outputf
            ###calc loss  
            lossx2=outputMat[x,:]-yMat[x,:]
            lossx2=lossx2*lossx2.T
            lossx2=lossx2[0,0]*0.5
            ######difference between loss
            loss12=(lossx1-lossx2)/eps*2.0
            #######check: compare numgrad with derivative grad
            if abs(loss12-gradc[c,dim])>0.001:
                print c,dim,'gradc1-gradc2',loss12-gradc[c,dim]    
    
def gradCheckerW(x):#x index not xvec
    global gradc,gradw,gradb,gradbb
    global hMat,hhMat,outputMat
    global Cmat,Wmat,bmat,bbmat
    global outputMat,yMat,dataMat
    
    eps=1.0
    for w in range(nfilter):#20
        for dim in range(filterdd):#9x9
            ###################calc loss postive
            #####change one dim of one paravec :theta vec postive=theta vec + [1,0,0,0,..]
            Wmat[w,dim]=Wmat[w,dim]+eps
            x16=forward(x)#to get h outputf
            ###calc loss  
            lossx1=outputMat[x,:]-yMat[x,:]
            lossx1=lossx1*lossx1.T
            lossx1=lossx1[0,0]*0.5
            ##################calc loss negative
            #####theta vec negative=theta vec - [1,0,0,0,..]
            Wmat[w,dim]=Wmat[w,dim]-2.0*eps
            x16=forward(x)#to get h outputf
            ###calc loss  
            lossx2=outputMat[x,:]-yMat[x,:]
            lossx2=lossx2*lossx2.T
            lossx2=lossx2[0,0]*0.5
            ######difference between loss
            loss12=(lossx1-lossx2)/eps*2.0
            #######check: compare numgrad with derivative grad
            if abs(loss12-gradw[w,dim])>0.001:
                print w,dim,'gradc1-gradc2',loss12-gradw[w,dim]  

   

##########################################
def vec2mat(vec,nhang,nlie): #input vec 1x16 ouput matric 4x4
    if nhang!=nlie:print 'num hang must = num lie'
    n=nhang#for example :1x16->4x4
    Mat=np.mat(np.zeros((n,n)))
    for hang in range(n):
        for lie in range(n):
            pos=n*hang+lie
            Mat[hang,lie]=vec[0,pos]
    return Mat
    
    
def shuffleObs():
    global dataMat
    num,dim=np.shape(dataMat) #1394 piece of obs
    order=range(num)[:]  #0-100  for loss calc,101...for train obs by obs ///not work. must use whole set to train
    random.shuffle(order)
    #####
    
    batchList=[]
    for batch in range(len(order)/nbatch):
        batchList.append(order[batch*nbatch:batch*nbatch+nbatch])
    
    return batchList
    
    

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
initialH()
initialPara()
initialErr()
initialGrad()
#######
'''
x16=forward(1)
calcGrad(1,x16)
gradCheckerC(1)
gradCheckerW(1)

#####
'''
for ep in range(epoch):
    batchList=shuffleObs()
    alpha/=2.0
    for batch in batchList[:]: #obs=x index not vec
        initialGrad() #accumulated 10 obs grad in one batch
        for obs in batch:
            x16=forward(obs)
            loss=calcLoss()#loss calc with outputf and old para
            calcGrad(obs,x16)
        #######
        divideNormalizeGrad()
        updatePara()
    
    print  'epoch %d loss %f'%(ep,loss)


###output

#####output para w m n c ,b
global Cmat,Wmat,bmat,bbmat

 
outfile1 = "C.txt"
outfile2 = "W.txt"

outfile3 = "B.txt"
outfile4 = "BB.txt"
 

outPutfile=open(outPath+'/'+outfile1,'w')
n,m=np.shape(Cmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Cmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
##
outPutfile=open(outPath+'/'+outfile2,'w')
n,m=np.shape(Wmat)
for i in range(n):
    for j in range(m):
        outPutfile.write(str(Wmat[i,j]))
        outPutfile.write(' ')
    outPutfile.write('\n')
    
outPutfile.close()
###
outPutfile=open(outPath+'/'+outfile3,'w')
n,m=np.shape(bmat)

for j in range(m):
    outPutfile.write(str(bmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()
## 
outPutfile=open(outPath+'/'+outfile4,'w')
n,m=np.shape(bbmat)

for j in range(m):
    outPutfile.write(str(bbmat[0,j]))
    outPutfile.write(' ')
outPutfile.write('\n')
    
outPutfile.close()


 


 
    
         
    
    
    







    
    
