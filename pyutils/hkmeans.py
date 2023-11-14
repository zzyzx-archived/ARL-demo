'''
	H_KMeans: hierarchical KMeans tree

	Attributes:
		refer to attributes in reset()

	Usage:
		clf = H_KMeans(max_clusters, min_n_samples, th_purity)
		pred = clf.fit(data,y)
		display(clf) # shows training results

	criterion: the splitting criterion
		mse: MSE
		var: Variance
		reg: Regresssion error
		bal: Balance = 1-min(node number)/max(node number)

	initial: initialization method
		ori: k-means++
		far: Farthest pair first
		sta: statistics-base
'''

import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from queue import Queue

from .kmeans_utils import *


class H_KMeans(object):
    def __init__(self, max_clusters=None, min_n_samples=None, th_purity=0, initial='ori'):
        self.reset(max_clusters, min_n_samples, th_purity, None)
        self.initial = initial

    def reset(self, max_clusters, min_n_samples, th_purity, criterion):
        # centroids in the tree
        self.n_nodes = 0
        self.centroids = []
        # minimum size of a splitable cluster
        if (min_n_samples is None):
            self.min_n_samples = 1
        else:
            self.min_n_samples = min_n_samples
        # labels
        self.labels = None  # raw prediction - clusterId
        self.class_labels = None  # majority class of each leaf node
        # for regression
        self.reg = None  # dict to store LSR for all leaf nodes
        # arrays to store the tree structure
        if (max_clusters is None):
            self.father = None
            self.leftC = None
            self.rightC = None
        else:
            self.father = -np.ones(max_clusters).astype(np.int32)
            self.leftC = -np.ones(max_clusters).astype(np.int32)
            self.rightC = -np.ones(max_clusters).astype(np.int32)
        # leaf node list
        self.th_purity = th_purity
        self.leaf_list = list()
        self.leaf_mse = list()
        # maximum number of clusters
        self.max_clusters = max_clusters
        # train flag
        self.trained = False
        self.criterion = criterion

    def cluster_splitting(self, data, y, clusterId, num_init=5, criterion=['mse']):
        '''
		Split one cluster into two clusters
		'''

        # samples in this cluster
        dataIdx = np.arange(len(self.labels))[self.labels == clusterId]
        tmp_data = data[dataIdx]
        if (y is None):
            tmp_y = None
        else:
            tmp_y = y[dataIdx]
        # splitting
        clf = kmeans_multiRuns(num_init, tmp_data, tmp_y, criterion, initial=self.initial)

        centroids = clf.cluster_centers_
        self.centroids.append(centroids[0])
        self.centroids.append(centroids[1])
        self.n_nodes += 2
        # update labels of samples
        clusterId1 = self.n_nodes - 2
        clusterId2 = self.n_nodes - 1
        self.update_labels(dataIdx, clf.labels_, clusterId1, clusterId2)
        return clusterId1, clusterId2

    def regression_train(self, data, y):
        self.reg = dict()
        for clusterId in self.leaf_list:
            dataIdx = (self.labels == clusterId)
            tmp_gt_labels = y[dataIdx]
            reg = LSR().fit(data[dataIdx], tmp_gt_labels)
            self.reg[str(int(clusterId))] = reg

    def regression_test(self, data, clusterId):
        if (len(data.shape) < 2):
            data = data.reshape((1, -1))
        pred = self.reg[str(int(clusterId))].predict(data)
        pred[pred < 0] = 0
        pred[pred > 1] = 1
        return pred

    def fit(self, data, y=None, num_init=5, criterion=['mse']):
        self.reset(2 * data.shape[0] + 1, self.min_n_samples, self.th_purity, criterion)

        self.labels = np.zeros(data.shape[0]).astype(np.int32)

        self.centroids.append(np.mean(data, axis=0))
        self.n_nodes += 1

        q = Queue(maxsize=self.max_clusters)
        q.put(0)
        cnt = 0
        while (not q.empty() and cnt < data.shape[0]):
            clusterId = q.get()
            clusterId1, clusterId2 = self.cluster_splitting(data, y, clusterId, num_init, criterion)
            # judge whether to split new clusters
            if (self.split_or_not(data, clusterId1)):
                q.put(clusterId1)
            if (self.split_or_not(data, clusterId2)):
                q.put(clusterId2)
            # update the tree
            self.maintain_tree(clusterId, clusterId1, clusterId2)
            cnt += 1
            if (cnt >= data.shape[0]):
                print('Warning: wrong splitting times')

        self.centroids = np.array(self.centroids)
        self.trained = True

        if (y is not None and isinstance(y[0], (int, np.int64, np.int32))):
            self.label_mapping(y)

        if ('reg' in criterion and y is not None):
            self.regression_train(data, y)

        return

    def label_mapping(self, y):
        '''
		Find out majority class for all leaf nodes
		'''
        self.class_labels = dict()
        for i in range(len(self.leaf_list)):
            idx = self.labels == self.leaf_list[i]
            tmp = y[idx]
            mode_result = stats.mode(tmp)
            self.class_labels[str(int(self.leaf_list[i]))] = mode_result[0][0]

    def update_labels(self, dataIdx, pred_labels, clusterId1, clusterId2):
        self.labels[dataIdx[pred_labels == 0]] = clusterId1
        self.labels[dataIdx[pred_labels == 1]] = clusterId2

    def split_or_not(self, data, clusterId):
        dataIdx = self.labels == clusterId
        mse_value = MSE(data[dataIdx])
        if (mse_value <= self.th_purity or np.sum(dataIdx) <= self.min_n_samples):
            self.leaf_list.append(clusterId)
            self.leaf_mse.append(mse_value)
            return False
        else:
            return True

    def maintain_tree(self, rootId, leftId, rightId):
        self.leftC[rootId] = leftId
        self.rightC[rootId] = rightId
        self.father[leftId] = rootId
        self.father[rightId] = rootId

    def find_cluster(self, pt):
        '''
		For test stage, find which cluster the point falls into.
		'''

        clusterId = 0
        while (self.leftC[clusterId] != -1 or self.rightC[clusterId] != -1):
            leftId = self.leftC[clusterId]
            rightId = self.rightC[clusterId]
            left_centroid = self.centroids[leftId]
            right_centroid = self.centroids[rightId]
            if (np.linalg.norm(pt - left_centroid) < np.linalg.norm(pt - right_centroid)):
                clusterId = leftId
            else:
                clusterId = rightId
        return clusterId

    def predict(self, data):
        assert (len(data.shape) > 1), 'single sample should be of size (1,num_feat)'

        centroids = self.centroids[np.array(self.leaf_list, dtype=np.int32)]
        dist2 = euclidean_distances(data, centroids, squared=True)
        idx = np.argmin(dist2, axis=1)
        pred_raw = np.array([self.leaf_list[v] for v in idx], dtype=np.int32)

        pred = None
        if 'var' in self.criterion and self.class_labels is not None:
            pred = [self.class_labels[str(v)] for v in pred_raw]
            pred = np.array(pred).astype(np.int32)
        elif 'reg' in self.criterion and self.reg is not None:
            pred = np.zeros((data.shape[0],)).astype(np.float32)
            for i in range(data.shape[0]):
                tmp = self.regression_test(data[i], pred_raw[i])
                pred[i] = tmp[0]

        return pred_raw, pred

    def save(self, filename):
        par = {}
        # centroids in the tree
        par['n_nodes'] = self.n_nodes
        par['centroids'] = self.centroids
        # minimum size of a splitable cluster
        par['min_n_samples'] = self.min_n_samples
        # labels
        par['labels'] = self.labels  # raw prediction
        # idx tree
        par['father'] = self.father
        par['leftC'] = self.leftC
        par['rightC'] = self.rightC
        # pure and non-pure leaf node list
        par['th_purity'] = self.th_purity
        par['leaf_list'] = self.leaf_list
        par['leaf_mse'] = self.leaf_mse
        # maximum number of clusters
        par['max_clusters'] = self.max_clusters
        # train flag
        par['trained'] = self.trained

        fw = open(filename + '.pkl', 'wb')
        pickle.dump(par, fw)
        fw.close()
        return

    def load(self, filename):
        par = pickle.load(open(filename, 'rb'))
        # centroids in the tree
        self.n_nodes = par['n_nodes']
        self.centroids = par['centroids']
        # minimum size of a splitable cluster
        self.min_n_samples = par['min_n_samples']
        # labels
        self.labels = par['labels']
        # idx tree
        self.father = par['father']
        self.leftC = par['leftC']
        self.rightC = par['rightC']
        # pure and non-pure leaf node list
        self.th_purity = par['th_purity']
        self.leaf_list = par['leaf_list']
        self.leaf_mse = par['leaf_mse']
        # maximum number of clusters
        self.max_clusters = par['max_clusters']
        # train flag
        self.trained = par['trained']
        return


class Sub_kmeans(object):
    def __init__(self, initial=['ori']):
        self.initial = initial

    def fit(self, con_train, y=None, num_init=5, criterion=['mse']):
        # Subindex: Feature index of each Subspace
        self.criterion = criterion
        Subindex = self.Subindex
        # Build classifier fit()
        Clfs = []
        for i in range(len(Subindex)):
            data = con_train[:, Subindex[i]]
            clf = kmeans_multiRuns(num_init, data, y=None, criterion=self.criterion, initial=self.initial)
            Clfs.append(clf)
        self.Clfs = Clfs
        self.node, self.raw = self.predict(con_train)
        return

    def predict(self, con_train):
        # do classification
        Clfs = self.Clfs
        Subindex = self.Subindex
        preds = []
        for i in range(len(Clfs)):
            data = con_train[:, Subindex[i]]
            pred = Clfs[i].predict(data)
            preds.append(pred)
        preds = np.array(preds)

        # bit to num
        preds_node = np.zeros((preds.shape[1]))
        for i in range(preds_node.shape[0]):
            tmp = np.str(preds[:, i]).replace(' ', '').replace('[', '').replace(']', '')
            preds_node[i] = int(tmp, 2)
        return preds_node, preds

    # Build sub dimension and do clustering on each sub dimension
    def Cluster_assign(self, Subnum, con_train, par_saab, SubDC_F=True):
        self.SubDC_F = SubDC_F
        # return: list containing Cluster assignment
        # Subnum: Number of sub-dimension
        # SubDC:  Whether separate DC feature
        if Subnum < 2: return
        # Get layer number
        layerIndex = np.arange(0, con_train.shape[1], 1)
        layerNum = []
        for i in range(par_saab['depth']):
            layerNum.append(par_saab['SaabArgs'][i]['num_AC_kernels'])

        # Assign DC
        if SubDC_F:
            SubDC = [0]
            for i in range(1, par_saab['depth']):
                SubDC.append(np.sum(layerNum[:i]))
            layerIndex = np.delete(layerIndex, SubDC)
            Subnum -= 1

        # Assign AC
        ClusNum = int(np.ceil(len(layerIndex) / Subnum))
        actualNum = len(layerIndex) / Subnum
        dif = 0
        SubNumL = []
        # [list(layerIndex[i:i+int(ClusNum)]) for i in range(0, len(layerIndex), int(ClusNum))]
        for i in range(Subnum):
            dif += ClusNum - actualNum
            SubNumL.append(ClusNum - int(dif))
            dif -= int(dif)
        # assign feature
        SubIndex = []
        end = 0
        for i in range(Subnum):
            start = end
            end = start + SubNumL[i]
            SubIndex.append(list(layerIndex[start:end]))

        if SubDC_F: SubIndex = [SubDC] + SubIndex
        self.Subindex = SubIndex
        return self.Subindex

    def display(self):
        print('--------------------------')
        print('Separate DC:', self.SubDC_F)
        print('Initializtion:', self.initial)
        print('Splitting criterion:', self.criterion)
        print('The number of subspace:', len(self.Subindex))
        print('The indexes of Subspace are', self.Subindex)
        b, node_sta = np.unique(self.node, return_counts=True)
        raw_sta = []
        for i in range(self.raw.shape[0]):
            _, tmp = np.unique(self.raw[i, :], return_counts=True)
            raw_sta.append(tmp)
        print('Number of nodes:', len(b))
        print('The statistic of training node is:')
        print(list(b.astype('int8')))
        print(list(node_sta))
        print('The splitting number of each Subspace is:')
        print(np.array(raw_sta))
        print('--------------------------')
