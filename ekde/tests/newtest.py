#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:35:43 2022

@author: frem
"""

from ekde import KDE
import numpy as np
from matplotlib import pyplot as plt

x_train = np.random.normal(0, 1, 10**3)

kde = KDE(kernel='gaussian').fit(x_train[:,None])
x_test = np.linspace(-5,5,100)
y_test = kde.predict(x_test[:,None])

plt.plot(x_test, y_test)

print(kde.predict(np.array([1,1,1,1,0,0,0])[:,None]))