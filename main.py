import random
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import h5py
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score,  matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(1)
np.random.seed(1)
Feature_Size = 420
num_class = 2
unit_size = 420
min_max_scaler = preprocessing.MinMaxScaler()

def GAN():

    LR_G = 0.0002
    LR_D = 0.0002

    G = nn.Sequential(
        nn.Linear(Feature_Size, 210),
        nn.ReLU(),
        nn.Linear(210, unit_size)
    )
    D = nn.Sequential(
        nn.Linear(unit_size, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

    optimizer_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LR_D)

    plt.ion()

    path1 = 'dataset/DRK_train.h5'
    # load train data
    with h5py.File(path1, 'r') as hf:
        posData = torch.from_numpy(np.asarray(hf[str(0)])).float()

    # train generator with small class data (specific rate)
    # return generator
    for step in range(1000):
        # data_0, data_1, data = load_data(path, 0)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # posData = min_max_scaler.fit_transform(posData)
        G_idea = torch.randn(posData.shape[0], Feature_Size)
        G_generate = G(G_idea).detach()
        pro_atrist0 = D(posData)
        pro_atrist1 = D(G_generate)
        G_loss = -1 / torch.mean(torch.log(1. - pro_atrist1))
        D_loss = -torch.mean(torch.log(pro_atrist0) + torch.log(1 - pro_atrist1))

        G_loss.backward(retain_graph=True)
        optimizer_G.step()

        D_loss.backward(retain_graph=True)
        optimizer_D.step()

        print("[epoch %d][D loss:%f][G loss: %f]" % (step, D_loss, G_loss))
    return G

# my down sampler
# random
def random_sampler(data_0, scale):
    index = random.sample(range(0, 280), scale)
    data_0 = np.asarray(data_0)
    data_0_sample = data_0[index][:]
    return data_0_sample

# classfication
# return: roc spec sen
def decision_tree(x, y, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)
    y_pre = clf.predict(x_test)
    conf_matrix = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pre)
    spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
    sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
    # roc1 = (spec + sen) / 2
    auc = roc_auc_score(y_test, y_pre, average='micro')
    acc = accuracy_score(y_test, y_pre)
    mcc = matthews_corrcoef(y_test, y_pre)
    g_mean = geometric_mean_score(y_test, y_pre, average='micro')
    return auc, spec, sen, acc, g_mean, mcc

class SVM:
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.ones(self.m)
        self.computer_product_matrix()
        self.C = 1.0
        self.create_E()

    #KKT
    def judge_KKT(self, i):
        y_g = self.function_g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def computer_product_matrix(self):
        self.product_matrix = np.zeros((self.m,self.m)).astype(np.float64)
        for i in range(self.m):
            for j in range(self.m):
                if self.product_matrix[i][j]==0.0:
                    self.product_matrix[i][j]=self.product_matrix[j][i]= self.kernel(self.X[i], self.X[j])

    def kernel(self, x1, x2):
        if self._kernel == 'linear':
            return np.dot(x1,x2)
        elif self._kernel == 'poly':
            return (np.dot(x1,x2) + 1) ** 2
        return 0

    def create_E(self):
        self.E=(np.dot((self.alpha * self.Y),self.product_matrix)+self.b)-self.Y

    def function_g(self, i):
        return self.b+np.dot((self.alpha * self.Y),self.product_matrix[i])

    def select_alpha(self):
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self.judge_KKT(i):
                continue
            E1 = self.E[i]
            if E1 >= 0:
                j =np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i, j

   def clip_alpha(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha
    def Train(self, features, labels):
        self.init_args(features, labels)
        for t in range(self.max_iter):
            i1, i2 = self.select_alpha()

            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = self.clip_alpha(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.create_E()

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else 0

    def score(self, X_test, y_test):
        y_pre = []
        for i in range(len(X_test)):
            pr = self.predict(X_test[i])
            y_pre.append(pr)
        conf_matrix = sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pre)
        spec = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])  # pos label 0
        sen = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # pos label 0
        roc1 = (spec + sen) / 2
        return roc1, spec, sen

def KeyNegdata(posData, negData):
    pD = posData.tolist()
    nD = negData.tolist()
    
    softmax = torch.nn.Softmax()
    posNumber = len(pD)
    posNumber2 = 2 * posNumber
    negSoftmax = []
    negSoftmax2 = []
    negFin = []
    for i in range(len(nD)):
        sim = cosine_similarity( np.array([nD[i]]),nD)
        softmax_sim =  softmax(torch.tensor(sim[0]))
        negSoftmax.append(softmax_sim[i])
   
    arr_min = heapq.nsmallest(posNumber2, negSoftmax)
    index_min = map(negSoftmax.index, arr_min)
    list_min = list(index_min)

    for i in range(posNumber2):
        allData = np.concatenate((np.array([nD[list_min[i]]]), pD), axis=0)
        sim = cosine_similarity( np.array([nD[list_min[i]]]),allData)
        softmax_sim2 =  softmax(torch.tensor(sim[0]))
        negSoftmax2.append(softmax_sim2[0])

    arr_max = heapq.nlargest(posNumber, negSoftmax2)
    index_max = map(negSoftmax2.index, arr_max)
    list_max = list(index_max)

    for i in range(posNumber):
        j = list_max[i]
        negFin.append(np.array([nD[list_min[j]]]))

    return np.array(negFin)

if __name__ == '__main__':
    # get generator
    G = GAN()
    finale_result = []
    for j in range(1):
        scale = 116
        result_single_scale_auc = []
        result_single_scale_sen = []
        result_single_scale_spec = []
        result_single_scale_acc = []
        result_single_scale_gmean = []
        result_single_scale_mcc = []

        # 10 train 10 test
        for i in range(10):

            path1 = 'dataset/DRK_train.h5'
            path2 = 'dataset/DRK_val.h5'

            data_set = np.zeros((0, Feature_Size + 1))

            # load train data
            with h5py.File(path1, 'r') as hf:
                # for i in range(num_class):
                #     data = np.asarray(hf[str(i)])
                #     # add label
                #     label = np.asarray([i for m in range(len(data))])
                #     label = label[np.newaxis, :]
                #     data = np.c_[data, label.T]
                #     data_set = np.concatenate((data_set, data), axis=0)
                # label = np.asarray([1 for m in range(len(negData))])
                # label = label[np.newaxis, :]
                # label = label.T
                posData = torch.from_numpy(np.asarray(hf[str(0)])).float()
                negData = torch.from_numpy(np.asarray(hf[str(1)])).float()
            # train_data, train_label = data_set[:, :-1], data_set[:, -1]
            print(posData.size(), negData.size())

            # posData = min_max_scaler.fit_transform(posData)
            # negData = min_max_scaler.fit_transform(negData)

            with h5py.File(path2, 'r') as hf:
                for i in range(num_class):
                    data = np.asarray(hf[str(i)])
                    # add label
                    label = np.asarray([i for m in range(len(data))])
                    label = label[np.newaxis, :]
                    data = np.c_[data, label.T]
                    data_set = np.concatenate((data_set, data), axis=0)
            test_data, test_label = data_set[:, :-1], data_set[:, -1]

            # test_data = min_max_scaler.fit_transform(test_data)

            # load train data
            # data_0, data_1, data = load_data(path, i)
            # print(data_0.shape, data_1.shape, data.shape)

            # make train data
            ## gan-based oversampling
            input = torch.randn(int(scale-posData.shape[0]), Feature_Size)
            output = G(input)
            posData = np.concatenate([posData, output.data.numpy()], axis=0)
            # print(posData)

            # attention-based undersampling
            negData = KeyNegdata(posData, negData)
            negData = np.reshape(negData,(-1,420))

            ## k-means undersampling
            # model = KMeans(n_clusters=scale, max_iter=100)
            # model.fit(negData)
            # negData = model.cluster_centers_

            # print(negData)

            # set label
            label_0 = np.zeros(posData.shape[0])
            label_0 = label_0.reshape((posData.shape[0], 1))
            posData = np.concatenate([posData, label_0], axis=1)
            # print(data_1.shape)

            # negData = random_sampler(negData, scale)  # big class down sample

            # set label
            label_1 = np.ones(negData.shape[0])
            label_1 = label_1.reshape((negData.shape[0], 1))
            negData = np.concatenate([negData, label_1], axis=1)
            data = np.concatenate((posData, negData), axis=0)
            # print(posData.shape, negData.shape)
            np.random.shuffle(data)
            data = np.asarray(data)
            X, Y = data[:, 0:-1], data[:, -1]

            # check data shape
            print(X.shape, Y.shape)
            # cls

            # svm = SVM(max_iter=10)
            # svm.Train(X, Y)
            # #
            # roc, spec, sen = svm.score(test_data, test_label)
            # print(roc,spec,sen)

            auc, spec, sen, acc, g_mean, mcc = decision_tree(X, Y, test_data, test_label)
            result_single_scale_auc.append(auc)
            result_single_scale_spec.append(spec)
            result_single_scale_sen.append(sen)

        finale_result.append([max(result_single_scale_sen),
                              max(result_single_scale_spec),
                              max(result_single_scale_auc),
                              ])


    print(finale_result)
