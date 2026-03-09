# -*- coding: utf-8 -*-

import numpy as np
import ekde

#%%
folder = "/home/frem/Work/LUCC/data/esnet/regions/sud_gresivaudan_artif_transitions/output/"

X = np.load(folder+'X.npy')
V = np.load(folder+'V.npy')

#%%
kde = ekde.KDE(q=11,
               bounds=[
                    (0,'left'),
                    (1,'left'),
                    (2,'left'),
                   ],
               verbose=1)
kde.fit(X)

#%%
from time import time
st = time()
f = kde.predict(X)
print(time()-st)

#%%
i = 3325
print(X[i])
print(f[i])
