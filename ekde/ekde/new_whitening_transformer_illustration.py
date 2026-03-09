#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:53:42 2022

@author: frem
"""

# this example is based on the ZCA the ZCA-Mahalanobis whitening transformation procedure (ZCA stands for ``zero-phase components analysis''). See \Kessy et al 2018 for more details.

# @Article{Kessy2018,
#   author    = {Kessy, A. and Lewin, A. and Strimmer, K.},
#   journal   = {The American Statistician},
#   title     = {Optimal {Whitening} and {Decorrelation}},
#   year      = {2018},
#   IGNOREissn      = {0003-1305},
#   IGNOREmonth     = oct,
#   number    = {4},
#   pages     = {309--314},
#   volume    = {72},
#   IGNOREabstract  = {Whitening, or sphering, is a common preprocessing step in statistical analysis to transform random variables to orthogonality. However, due to rotational freedom there are infinitely many possible whitening procedures. Consequently, there is a diverse range of sphering methods in use, for example, based on principal component analysis (PCA), Cholesky matrix decomposition, and zero-phase component analysis (ZCA), among others. Here, we provide an overview of the underlying theory and discuss five natural whitening procedures. Subsequently, we demonstrate that investigating the cross-covariance and the cross-correlation matrix between sphered and original variables allows to break the rotational invariance and to identify optimal whitening transformations. As a result we recommend two particular approaches: ZCA-cor whitening to produce sphered variables that are maximally similar to the original variables, and PCA-cor whitening to obtain sphered variables that maximally compress the original variables.},
#   file      = {:storage/LUCC/Kessy2018.pdf:PDF},
#   groups    = {Whitening Transformation},
#   IGNOREkeywords  = {CAR score, CAT score, Cholesky decomposition, Decorrelation, Principal components analysis, Whitening, ZCA-Mahalanobis transformation},
#   IGNOREpublisher = {Taylor \& Francis},
#   IGNOREurldate   = {2021-09-30},
#   doi       = {10.1080/00031305.2016.1277159}
# }

#%%
import numpy as np
from scipy.stats import multivariate_normal

cov = np.array([[1, 0.2, 0.3],
                [0.2,0.8,-0.1],
                [0.3,-0.1,1.2]])
mean = np.array([0.1,-0.2, 0.3])

n = 10000
X = multivariate_normal.rvs(mean=mean, cov=cov, size=n)
mu = X.mean(axis=0)
X_bar = X - mu
C = np.cov(X_bar.T)
print(C)

L, V = np.linalg.eig(C) 
L = np.diag(L)
print(L)
print(V)

L_inv_squared = np.diag(np.diag(L)**(-0.5))
# print(L_inv_squared)
W = np.dot(np.dot(V, L_inv_squared),V.T)
print(W)
# print(np.dot(np.dot(V, L), V.T))

X_star = X_bar @ W

print(np.cov(X_star.T))