#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:28:25 2022

@author: frem
"""

# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import ekde
from time import time

import fastkde
import KDEpy

#%%
n = 10**6
d = 5

np.random.seed(42)
X = stats.multivariate_normal.rvs(mean=np.zeros(d), cov=np.diag(np.ones(d)), size=n)

np.random.seed(51)
Y = stats.multivariate_normal.rvs(mean=np.zeros(d), cov=np.diag(np.ones(d)), size=n)

np.random.seed(None)

kde_ekde = ekde.KDE(q=31, verbose=1).fit(X)

h = ekde.base.scotts_rule(X)

#%%
from sklearn.neighbors import KNeighborsRegressor

class KDEpy_FFTKDE():
    def __init__(self, 
                 h,
                 s,
                 q,
                 kernel='gaussian'):
        self.h = h
        self.s = s
        self.q = q
        self.kernel = kernel
        
        self.kde = KDEpy.FFTKDE(kernel=self.kernel, bw=self.h)
        
    def fit(self, X):
        self.kde.fit(X)
        
        self.n_grid = int(np.max(X.max(axis=0) - X.min(axis=0)) / (self.h * self.s / self.q))
        
    def predict(self, X):
        
        X_grid, f_grid = self.kde.evaluate(self.n_grid)
        
        knr = KNeighborsRegressor(n_neighbors=1).fit(X_grid, f_grid)
        
        return(knr.predict(X))
    
from fastkde import fastKDE
class FastKDE_process():
    def __init__(self,
                 h,
                 s,
                 q):
        self.h = h
        self.s = s
        self.q = q
        
    def fit(self, X):
        self.data = X
        
        self.n_grid = int(np.max(X.max(axis=0) - X.min(axis=0)) / (self.h * self.s / self.q))
        
        self.n_grid = 2**(int(np.log2(self.n_grid)) + 1) + 1
    
    def predict(self, X):
        
        axes = [np.linspace(x.min(), x.max(), self.n_grid) for x in X.T]
        
        fkde = fastKDE.fastKDE(data=self.data.T,
                               axes=axes)
        
        axes = np.meshgrid(*axes)
        X_grid = np.vstack([a.flat for a in axes]).T
        
        knr = KNeighborsRegressor(n_neighbors=1).fit(X_grid, fkde.pdf.flat)
        
        return(knr.predict(X))

#%%
st = time()
fftkde = KDEpy_FFTKDE(h=h,
                      s=3,
                      q=31)
fftkde.fit(X)
f_kdepy_fft = fftkde.predict(Y)
time_kdepy = time()-st
print(time_kdepy)
#%%
fkde = FastKDE_process(h=h, s=3, q=31)
fkde.fit(X)
f_fastkde = fkde.predict(Y)

#%%
kde_ekde = ekde.KDE(q=31, verbose=1).fit(X)
f_ekde = kde_ekde.predict(Y)

#%%
print(np.max(f_ekde - stats.multivariate_normal.pdf(Y, mean=np.zeros(d), cov=np.diag(np.ones(d)))))