import numpy as np
from pyrwr.rwr import RWR


def simabs(SIM):
    for i in range(SIM.shape[0]):
        for j in range(SIM.shape[0]):
            if SIM[i][j] < 0:
                SIM[i][j] = abs(SIM[i][j])
    return SIM


def formart_DISandRNA(RNA_CIS, Disease_DIS):
    DIS = []
    RNA = []
    for i in range(Disease_DIS.shape[0]):
        for j in range(Disease_DIS.shape[0]):
            DIS_inner = []
            DIS_inner.append(i)
            DIS_inner.append(j)
            DIS_inner.append(Disease_DIS[i][j])
            DIS.append(DIS_inner)
    for i in range(RNA_CIS.shape[0]):
        for j in range(RNA_CIS.shape[0]):
            RNA_inner = []
            RNA_inner.append(i)
            RNA_inner.append(j)
            RNA_inner.append(RNA_CIS[i][j])
            RNA.append(RNA_inner)
    np.savetxt('../dataset/rwr/RWR_FORMART_DIS.csv', DIS)
    np.savetxt('../dataset/rwr/RWR_FORMART_CIS.csv', RNA)
    return DIS, RNA


def SimtoRWR(RNA_CIS, Disease_DIS, FLAGS):
    RNA_CIS = simabs(RNA_CIS)
    Disease_DIS = simabs(Disease_DIS)
    path_DIS = '../dataset/rwr/RWR_FORMART_DIS.csv'
    path_CIS = '../dataset/rwr/RWR_FORMART_CIS.csv'
    formart_DISandRNA(RNA_CIS, Disease_DIS)
    RWRRNA = np.array(RWR_Comp(path_CIS, RNA_CIS.shape[0], FLAGS, 'RSim'))
    RWRDIS = np.array(RWR_Comp(path_DIS, Disease_DIS.shape[0], FLAGS, 'DSim'))
    return RWRRNA, RWRDIS


def RWR_Comp(path, w, FLAGS, NAME):
    FEATURE = []
    rwr = RWR()
    rwr.read_graph(path, "directed")
    for i in range(w):
        r = rwr.compute(i, c=0.9)
        FEATURE.append(sorted(r, reverse=True))
    print(NAME + ' finish')
    return FEATURE


def FeatureNormalization(FEATURE, NAME):
    Max = np.max(FEATURE)
    Min = np.min(FEATURE)
    FEATURE_nor = []
    for factor in FEATURE:
        FEATURE_nor.append(MaxMinNormalization(factor, Max, Min))
    print(NAME + ' Normalization finish')
    return FEATURE_nor


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x
