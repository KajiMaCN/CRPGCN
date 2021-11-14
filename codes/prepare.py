import numpy as np
import pandas as pd
from tqdm import trange
from codes.rwr import SimtoRWR
from codes.pca import reduction


def Makeadj(AM):
    adj = []
    for i in trange(AM.shape[0]):
        for j in range(AM.shape[1]):
            adj_inner = []
            if AM[i][j] == 1:
                adj_inner.append(i + 1)
                adj_inner.append(j + 1)
                adj_inner.append(1)
                adj.append(adj_inner)
    return np.array(adj)


def heteg(SC, SD, AM):
    reSC = np.hstack((SC, AM))
    reSD = np.hstack((SD, AM.T))
    return reSC, reSD


def prepareData(FLAGS):
    # Reading data from disk
    AM = pd.read_csv('../dataset/AM.csv', header=None).values
    CS = pd.read_csv('../dataset/RNA_sim.csv', header=None).values
    DS = pd.read_csv('../dataset/Disease_sim.csv', header=None).values

    # Adjacency matrix transformation
    adj = Makeadj(AM)

    # Using RWR to calculate CRS and DRS
    CRS, DRS = SimtoRWR(CS, DS, FLAGS)

    # Matrix Splicing
    reSC, reSD = heteg(CRS, DRS, AM)

    # Matrix noise reduction and dimensionality reduction
    CF, DF = reduction(reSC, reSD, FLAGS)
    return adj, CS, DS, CF, DF, AM
