# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.array([0.1, 1.4, 1.6])
h = 0.3

y = np.linspace(-0.2, 2.6, 300)

f = np.zeros_like(y)

f_exact = norm.pdf(y, 0.0, 0.3) + 1* norm.pdf(y, 1.5, 0.3)
f_exact[y<0] = 0
f_exact /= f_exact.sum() * (y.max()-y.min()) / y.size

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)

plt.plot(y, f_exact, color='tab:blue', label='exact')

for xi in x:
    j = np.abs(y-xi) < h
    f_sub = np.zeros(y.size)
    f_sub = norm.pdf(y, xi, h)
    # f_sub[y<0] = 0
    f_sub *= 1 / x.shape[0]
    f += f_sub
    if xi == x[0]:
        plt.scatter(xi, -0.04, marker='|', s=70, color='gray', label='observations')
        plt.plot(y, f_sub, color='gray', ls='--', label='Gaussian kernels')
        
    else:
        plt.plot(y, f_sub, color='gray', ls='--')
        plt.scatter(xi, -0.04, marker='|', s=70, color='gray')

plt.plot(y, f, ls='-', color='tab:red', label='kde')
plt.xlabel('z')
plt.ylabel('probability distribution')
plt.title('(a) without correction')

print(f_exact.sum() * (y.max()-y.min()) / y.size)
print(f.sum() * (y.max()-y.min()) / y.size)

plt.subplot(1,2,2)
plt.plot(y, f_exact, color='tab:blue', label='exact')
f = np.zeros_like(y)

for xi in x:
    j = np.abs(y-xi) < h
    f_sub = np.zeros(y.size)
    f_sub = norm.pdf(y, xi, h)
    f_sub[y<0] = 0
    f_sub *= 1 / x.shape[0]
    
    c = norm.cdf(xi + (xi - 0), xi, h)
    f_sub /= c
    
    f += f_sub
    if xi == x[0]:
        plt.scatter(xi, -0.04, marker='|', s=70, color='gray', label='observations')
        plt.plot(y, f_sub, color='gray', ls='--', label='Gaussian kernels')
        
    else:
        plt.plot(y, f_sub, color='gray', ls='--')
        plt.scatter(xi, -0.04, marker='|', s=70, color='gray')

plt.plot(y, f, ls='-', color='tab:red', label='kde')
plt.legend(loc='lower right', bbox_to_anchor=(1.55, 0.58))
plt.xlabel('z')
# plt.ylabel('probability distribution')
print(f.sum() * (y.max()-y.min()) / y.size)
plt.title('(b) with correction')
plt.savefig('/home/frem/Work/LUCC/gisworknotes/Articles/LUCCProbabilitiesEstimation_EnvModSoft/figures/boundary_bias.pdf', bbox_inches='tight')