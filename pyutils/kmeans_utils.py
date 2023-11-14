import numpy as np
from sklearn.cluster import KMeans as kmeans
from sklearn.linear_model import LinearRegression as LSR
from sklearn.metrics.pairwise import cosine_similarity


def MSE(tmp_data):
    centroid = np.mean(tmp_data, axis=0)
    MSE_list = np.linalg.norm(tmp_data-centroid,axis=1)
    return np.mean(MSE_list**2)


def judgeClustering(data,y,clf,criterion):
    centroid1 = clf.cluster_centers_[0]
    centroid2 = clf.cluster_centers_[1]
    clst1 = data[clf.labels_==0]
    clst2 = data[clf.labels_==1]
    if(y is None):
        y1 = None
        y2 = None
    else:
        y1 = y[clf.labels_==0]
        y2 = y[clf.labels_==1]

    if(criterion=='mse'):
        mse1 = clst1.shape[0]*MSE(clst1-centroid1)
        mse2 = clst2.shape[0]*MSE(clst2-centroid2)
        value = mse1+mse2
    elif(criterion=='var'):
        var1 = clst1.shape[0]*np.var(y1)
        var2 = clst2.shape[0]*np.var(y2)
        value = var1+var2
    elif(criterion=='reg'):
        reg1 = LSR().fit(clst1,y1)
        err1 = np.sum((y1-reg1.predict(clst1))**2)
        reg2 = LSR().fit(clst2,y2)
        err2 = np.sum((y2-reg2.predict(clst2))**2)
        value = err1+err2
    elif(criterion=='bal'):
        NodeSize=[]
        NodeSize.append(np.sum(clf.labels_==0))
        NodeSize.append(np.sum(clf.labels_==1))
        value=1-np.min(NodeSize)/np.max(NodeSize)
    else:
        print('Not implemented yet :(')
        value = 0
    return value


def isBetter(data,y,km1,km2,criterion):
    # Return True if km1 is better than km2
    value1 = 0
    value2 = 0
    for v in criterion:
        tmp1 = judgeClustering(data,y,km1,v)
        tmp2 = judgeClustering(data,y,km2,v)
        if(tmp1+tmp2>1e-15):
            value1 += tmp1/(tmp1+tmp2)
            value2 += tmp2/(tmp1+tmp2)
    return value1<value2


def far_point(data,centroid):
    dist = np.linalg.norm(data-centroid,axis=1)
    indx=np.argmax(dist)
    point=data[indx]
    newdata=np.delete(data,indx,axis=0)
    return point,newdata


def ortho_point(data,centroid,PointPair):
    cos=cosine_similarity(data-centroid,np.array(PointPair))
    cos=np.abs(cos)
    cos=np.max(cos,axis=1)
    indx=np.argmin(cos)
    p3=data[indx]
    data=np.delete(data,indx,axis=0)
    return p3,data


class fpf(object):
    '''
    parameters:
        self.ratio: threshold of far set
    '''
    def __init__(self,data,ratio=0.2):
        self.ratio=ratio
        self.init_time=0
        self.data=data.copy()
        self.centroid=np.mean(self.data, axis=0)
        self.data_far=self.far_point_set(self.data,self.centroid)
    def get_init(self):
        if self.init_time==0:
            high=self.data_far.shape[0]
            ind=np.random.randint(0, high, 1)[0]
            p1=self.data_far[ind,:]
            self.data_far=np.delete(self.data_far,ind,axis=0)
            p2,self.data_far=far_point(self.data_far,p1)
            self.PointList=[[p1,p2]]
            self.init_time+=1
        else:
            L=len(self.PointList)
            PointPair=[]
            for i in range(L):
                PointPair.append(self.PointList[i][0]-self.centroid)
            p3,self.data_far=ortho_point(self.data_far,self.centroid,PointPair)
            p4,self.data_far=far_point(self.data_far,p3)
            self.PointList.append([p3,p4])
            self.init_time+=1
        pair=self.PointList[-1]
        output=np.vstack(pair)
        return output
    def far_point_set(self,data,centroid):
        dist = np.linalg.norm(data-centroid,axis=1)
        indSort=np.argsort(dist)
        num=int(self.ratio*len(dist))
        data_far=data[indSort[-num:],:]
        return data_far


class fpf2(object):
    '''
    parameters:
        self.ratio: threshold of far set
    '''
    def __init__(self,data,ratio=0.1):
        self.ratio=ratio
        self.init_time=0
        self.data=data.copy()
        self.centroid=np.mean(self.data, axis=0)
        self.data_far=self.far_point_set(self.data,self.centroid)
        self.cal_all()
    def get_init(self,index=0):
        dis=np.abs(self.PointList[:,0]-self.PointList[:,1])
        Sort=np.argsort(dis)
        Sort=Sort[::-1]
        ind=Sort[index]
        init=np.zeros((2,self.data.shape[1]))
        init[:,ind]=self.PointList[ind]

        return init
    def cal_all(self):
        PointLists=[]
        for i in range(self.data.shape[1]):
            PointLists.append(self.get_pair(self.data[:,i]))
        self.PointList=np.array(PointLists)
    def get_pair(self,Data):
        p1=np.min(Data)
        p2=np.max(Data)
        PointList=np.hstack(([p1],[p2]))
        return PointList
    #     return output
    def far_point_set(self,data,centroid):
        dist = np.linalg.norm(data-centroid,axis=1)
        indSort=np.argsort(dist)
        num=int(self.ratio*len(dist))
        data_far=data[indSort[-num:],:]
        return data_far


def statistic_base(data,PointNum):
    # get centroid and standard deviation
    assert (PointNum>=2), 'PointNum cannot be less than 2'
    dim=data.shape[1]
    centroid=np.mean(data, axis=0)
    stand=np.std(data,axis=0)

    #generate random vectors consist of +1 and -1
    Num=0
    while Num<(PointNum-1):
        vector=np.random.randn(PointNum-1,dim)
        vector[vector>=0]=1
        vector[vector<0]=-1
        vector,indc=np.unique(vector,return_index=True,axis=0)
        Num=vector.shape[0]


    #connect all 
    P=centroid+stand*vector
    P=np.vstack((centroid,P))
    return P


def kmeans_multiRuns(num_init,data,y=None,criterion=['mse'],initial=['ori']):
    assert (num_init>=1), 'num_init cannot be less than 1'
    best_clf = None
    rng = 2020

    if 'sta' in initial:
        for i in range(num_init):
            clf = kmeans(n_clusters=2,init=statistic_base(data,2),n_init=1,max_iter=300,tol=1e-4,
                        precompute_distances='auto',verbose=0,random_state=rng,copy_x=True,
                        n_jobs=None,algorithm='auto')
            clf.fit(data)
            if(best_clf is None or isBetter(data,y,clf,best_clf,criterion)):
                best_clf = clf
    if 'far' in initial:
        a=fpf(data)
        for i in range(num_init):
            clf = kmeans(n_clusters=2,init=a.get_init(),n_init=1,max_iter=300,tol=1e-4,
                        precompute_distances='auto',verbose=0,random_state=rng,copy_x=True,
                        n_jobs=None,algorithm='auto')
            clf.fit(data)
            if(best_clf is None or isBetter(data,y,clf,best_clf,criterion)):
                best_clf = clf
    if 'far2' in initial:
        a=fpf2(data)
        clf = kmeans(n_clusters=2,init=a.get_init(),n_init=1,max_iter=300,tol=1e-4,
                    precompute_distances='auto',verbose=0,random_state=rng,copy_x=True,
                    n_jobs=None,algorithm='auto')
        clf.fit(data)
        if(best_clf is None or isBetter(data,y,clf,best_clf,criterion)):
            best_clf = clf
    if 'ori' in initial:
        for i in range(num_init):
            clf = kmeans(n_clusters=2,init='k-means++',n_init=1,max_iter=300,tol=1e-4,
                        precompute_distances='auto',verbose=0,random_state=rng,copy_x=True,
                        n_jobs=None,algorithm='auto')
            clf.fit(data)
            if(best_clf is None or isBetter(data,y,clf,best_clf,criterion)):
                best_clf = clf

    return best_clf
