import numpy as np
# from intra_alignment import CORAL_map, GFK_map, PCA_map
# from label_prop import label_prop
import numpy as np
import pulp


def  label_prop(C, nt, Dct, lp="linear"):
    # Inputs:
    #  C      :    Number of share classes between src and tar
    #  nt     :    Number of target domain samples
    #  Dct    :    All d_ct in matrix form, nt * C
    #  lp     :    Type of linear programming: linear (default) | binary
    # Outputs:
    #  Mcj    :    all M_ct in matrix form, m * C

    Dct = abs(Dct)
    model = pulp.LpProblem("Cost minimising problem", pulp.LpMinimize)
    Mcj = pulp.LpVariable.dicts("Probability",
                                ((i, j) for i in range(C) for j in range(nt)),
                                lowBound=0,
                                upBound=1,
                                cat='Continuous')

    # Objective Function
    model += (
        pulp.lpSum([Dct[j, i] * Mcj[(i, j)] for i in range(C) for j in range(nt)])
    )

    # Constraints
    for j in range(nt):
        model += pulp.lpSum([Mcj[(i, j)] for i in range(C)]) == 1
    for i in range(C):
        model += pulp.lpSum([Mcj[(i, j)] for j in range(nt)]) >= 1

    # Solve our problem
    model.solve()
    pulp.LpStatus[model.status]
    Output = [[Mcj[i, j].varValue for i in range(C)] for j in range(nt)]

    return np.array(Output)

def get_cosine_dist(A, B):
    B = np.reshape(B, (1, -1))

    if A.shape[1] == 1:
        A = np.hstack((A, np.zeros((A.shape[0], 1))))
        B = np.hstack((B, np.zeros((B.shape[0], 1))))

    aa = np.sum(np.multiply(A, A), axis=1).reshape(-1, 1)
    bb = np.sum(np.multiply(B, B), axis=1).reshape(-1, 1)
    ab = A @ B.T

    # to avoid NaN for zero norm
    aa[aa == 0] = 1
    bb[bb == 0] = 1

    D = np.real(np.ones((A.shape[0], B.shape[0])) - np.multiply((1 / np.sqrt(np.kron(aa, bb.T))), ab))

    return D


def get_ma_dist(A, B):
    Y = A.copy()
    X = B.copy()

    S = np.cov(X.T)
    try:
        SI = np.linalg.inv(S)
    except:
        print("Singular Matrix: using np.linalg.pinv")
        SI = np.linalg.pinv(S)
    mu = np.mean(X, axis=0)

    diff = Y - mu
    Dct_c = np.diag(diff @ SI @ diff.T)

    return Dct_c


def get_class_center(Xs, Ys, Xt, dist):
    source_class_center = np.array([])
    Dct = np.array([])
    for i in np.unique(Ys):
        sel_mask = Ys == i
        X_i = Xs[sel_mask.flatten()]
        mean_i = np.mean(X_i, axis=0)
        if len(source_class_center) == 0:
            source_class_center = mean_i.reshape(-1, 1)
        else:
            source_class_center = np.hstack((source_class_center, mean_i.reshape(-1, 1)))

        if dist == "ma":
            Dct_c = get_ma_dist(Xt, X_i)
        elif dist == "euclidean":
            Dct_c = np.sqrt(np.nansum((mean_i - Xt) ** 2, axis=1))
        elif dist == "sqeuc":
            Dct_c = np.nansum((mean_i - Xt) ** 2, axis=1)
        elif dist == "cosine":
            Dct_c = get_cosine_dist(Xt, mean_i)
        elif dist == "rbf":
            Dct_c = np.nansum((mean_i - Xt) ** 2, axis=1)
            Dct_c = np.exp(- Dct_c / 1)

        if len(Dct) == 0:
            Dct = Dct_c.reshape(-1, 1)
        else:
            Dct = np.hstack((Dct, Dct_c.reshape(-1, 1)))

    return source_class_center, Dct


def EasyTL(source_data, source_label, target_data, target_label, intra_align="coral", dist="euclidean", lp="linear"):
    # Inputs:
    #   Xs          : source data, ns * m
    #   Ys          : source label, ns * 1
    #   Xt          : target data, nt * m
    #   Yt          : target label, nt * 1
    # The following inputs are not necessary
    #   intra_align : intra-domain alignment: coral(default)|gfk|pca|raw
    #   dist        : distance: Euclidean(default)|ma(Mahalanobis)|cosine|rbf
    #   lp          : linear(default)|binary

    # Outputs:
    #   acc         : final accuracy
    #   y_pred      : predictions for target domain

    # Reference:
    # Jindong Wang, Yiqiang Chen, Han Yu, Meiyu Huang, Qiang Yang.
    # Easy Transfer Learning By Exploiting Intra-domain Structures.
    # IEEE International Conference on Multimedia & Expo (ICME) 2019.
    Xs = source_data.copy()
    Xt = target_data.copy()
    Ys = source_label.copy()
    Yt =  target_label.copy()

    C = len(np.unique(Ys))
    # if C > np.max(Ys):
    #     Ys += 1
    #     Yt += 1

    m = len(Yt)

    if intra_align == "raw":
        print('EasyTL using raw feature...')
    elif intra_align == "pca":
        print('EasyTL using PCA...')
        print('Not implemented yet, using raw feature')
    # Xs, Xt = PCA_map(Xs, Xt)
    elif intra_align == "gfk":
        print('EasyTL using GFK...')
        print('Not implemented yet, using raw feature')
        # Xs, Xt = GFK_map(Xs, Xt)
    elif intra_align == "coral":
        print('EasyTL using CORAL...')
        Xs = CORAL_map(Xs, Xt)

    _, Dct = get_class_center(Xs, Ys, Xt, dist)
    print('Start intra-domain programming...')
    Mcj = label_prop(C, m, Dct, lp)
    # y_pred = np.argmax(Mcj, axis=1) + 1
    y_pred = np.argmax(Mcj, axis=1)

    acc = np.mean(y_pred == Yt.flatten())

    return acc, Mcj


import numpy as np
import scipy
from sklearn.decomposition import PCA
import math


def GFK_map(Xs, Xt):
    pass


def gsvd(A, B):
    pass


def getAngle(Ps, Pt, DD):
    Q = np.hstack((Ps, scipy.linalg.null_space(Ps.T)))
    dim = Pt.shape[1]
    QPt = Q.T @ Pt
    A, B = QPt[:dim, :], QPt[dim:, :]
    U, V, X, C, S = gsvd(A, B)
    alpha = np.zeros([1, DD])
    for i in range(DD):
        alpha[0][i] = math.sin(np.real(math.acos(C[i][i] * math.pi / 180)))

    return alpha


def getGFKDim(Xs, Xt):
    Pss = PCA().fit(Xs).components_.T
    Pts = PCA().fit(Xt).components_.T
    Psstt = PCA().fit(np.vstack((Xs, Xt))).components_.T

    DIM = round(Xs.shape[1] * 0.5)
    res = -1

    for d in range(1, DIM + 1):
        Ps = Pss[:, :d]
        Pt = Pts[:, :d]
        Pst = Psstt[:, :d]
        alpha1 = getAngle(Ps, Pst, d)
        alpha2 = getAngle(Pt, Pst, d)
        D = (alpha1 + alpha2) * 0.5
        check = [round(D[1, dd] * 100) == 100 for dd in range(d)]
        if True in check:
            res = list(map(lambda i: i == True, check)).index(True)
            return res


def PCA_map(Xs, Xt):
    dim = getGFKDim(Xs, Xt)
    X = np.vstack((Xs, Xt))
    X_new = PCA().fit_transform(X)[:, :dim]
    Xs_new = X_new[:Xs.shape[0], :]
    Xt_new = X_new[Xs.shape[0]:, :]
    return Xs_new, Xt_new


def CORAL_map(Xs, Xt):
    Ds = Xs.copy()
    Dt = Xt.copy()

    cov_src = np.ma.cov(Ds.T) + np.eye(Ds.shape[1])
    cov_tar = np.ma.cov(Dt.T) + np.eye(Dt.shape[1])

    Cs = scipy.linalg.sqrtm(np.linalg.inv(np.array(cov_src)))
    Ct = scipy.linalg.sqrtm(np.array(cov_tar))
    A_coral = np.dot(Cs, Ct)

    Xs_new = np.dot(Ds, A_coral)

    return Xs_new