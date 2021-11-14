import numpy as np
import pandas as pd


def meanX(dataX):
    return np.mean(dataX, axis=0)


def variance(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))
    X1 = X - muAll
    variance = 1. / m * np.diag(X1.T * X1)
    return variance


def normalize(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))
    X1 = X - muAll
    X2 = np.tile(np.diag(X.T * X), (m, 1))
    XNorm = X1 / X2
    return XNorm


def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec = np.linalg.eig(covX)
    index = np.argsort(-featValue)
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = featVec.T[index[:k]]
        finalData = np.matmul(data_adjust, selectVec.T)
        reconData = np.matmul(finalData, selectVec) + average
        real_finalData = finalData.real
        real_reconData = reconData.real
    return real_finalData, real_reconData


def loaddata(datafile):
    return np.array(pd.read_csv(datafile, sep="\t", header=None)).astype(np.float)


def reduction(CS, DS, FLAGS):
    k_CS = int(CS.shape[1] * 0.8)
    k_DS = int(DS.shape[1] * 0.8)
    CS, reconMat = pca(CS, k_CS)
    DS, reconMat = pca(DS, k_DS)
    return CS, DS
