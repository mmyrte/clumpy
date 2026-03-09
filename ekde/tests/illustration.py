# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np

X = np.array([
              # [1.2,0.2],
              [1.3,1.2],
              [1.7,1.7],
              [2.4,1.7],
              [2.7,1.3],
              [2.5,2.2],
              [2.1,2.9],
               [2.8,1.6],
              # [3.2,1.3],
              [3.2,1.2],
              [3.8,2.4],
               [3.7, 2.3]
              # [3.7,2.3],
              # [4.6,2.5],
              # [4.2,3.2],
              ])

X += np.array([0, 1.0])

Y = np.array([[1.6, 2.2],
              # [2.1,1.2],
              # [3.8, 2.7],
              ])

fig, axs = plt.subplots(1,1)
axs.set_aspect('equal', 'box')

X_real = X.copy() + np.array([52,3])
Y_real = Y + np.array([52,3])

plt.scatter(X_real[:,0], X_real[:,1], label='observed points')
plt.scatter(Y_real[:,0], Y_real[:,1], label='point to estimate')
# plt.hlines(3.0, 53, 56, color='black', linewidth=3, label='border')
plt.legend(loc='lower left', bbox_to_anchor=(1,0.762))
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.savefig('/home/frem/Work/LUCC/gisworknotes/1_Articles/2_Calibration/figures/ekde_1.pdf', bbox_inches='tight')

#%%
fig, axs = plt.subplots(1,1)
axs.set_aspect('equal', 'box')
plt.scatter(X[:,0], X[:,1], s=20, label='observed')

plt.axis('off')

for x in [0,1,2,3,4, 5]:
    if x == 0:
        plt.vlines(x, 0, 5, color='gray', label='bin cells')
    else:
        plt.vlines(x, 0, 5, color='gray')
    if x < 5:
        plt.text(x+0.5, -0.5, x)

for y in [0,1,2,3,4, 5]:
    plt.hlines(y, 0, 5, color='gray')
    if y < 5:
        plt.text(-0.5, y+0.5, y)

# plt.hlines(0, 0, 5, color='black', linewidth=3, label='border')

plt.legend(loc='lower left', bbox_to_anchor=(1,0.715))

#%%
X_star = np.array([
                    [1,1],
                    # [2,0],
                    [2,1],
                   [2,2],
                   [3,1],
                   [3,2],
                   # [4,2],
                   # [4,3],
                   ])
X_star += np.array([0,1])

mu = np.array([2, 3,2,1,2])
# mu = mu / X.shape[0]

fig, axs = plt.subplots(1,1)
axs.set_aspect('equal', 'box')

plt.axis('off')



for x in [0,1,2,3,4, 5]:
    if x == 0:
        plt.vlines(x, 0, 5, color='gray', label='bin cells\nwith its $\\nu$ value noted\n($0$ if any)')
    else:
        plt.vlines(x, 0, 5, color='gray')
    if x < 5:
        plt.text(x+0.5, -0.5, x)

for y in [0,1,2,3,4, 5]:
    plt.hlines(y, 0, 5, color='gray')
    if y < 5:
        plt.text(-0.5, y+0.5, y)

# plt.hlines(0, 0, 5, color='black', linewidth=3, label='border')

from matplotlib import patches
from matplotlib.collections import PatchCollection

R = []
for x in X_star:
    R.append(patches.Rectangle(x, 1, 1))

colors = mu / mu.sum()
collection = PatchCollection(R, cmap=plt.cm.autumn, alpha=0.5)
collection.set_array(-colors)
# axs.add_collection(collection)

for i, x in enumerate(X_star):
    plt.text(x[0]+0.4, x[1]+0.37, round(mu[i],1))

plt.scatter(X[:,0], X[:,1], s=20, label='observed points', alpha=1)
# plt.text(5.5, 1, 'Bin values\nrefers to $\mathbf{\mu}$')
plt.legend(loc='lower left', bbox_to_anchor=(1,0.6))
plt.savefig('/home/frem/Work/LUCC/gisworknotes/1_Articles/2_Calibration/figures/ekde_2.pdf', bbox_inches='tight')


#%%
# fig, axs = plt.subplots(1,1)
# axs.set_aspect('equal', 'box')

# plt.axis('off')

# for x in [0,1,2,3,4, 5]:
#     if x == 0:
#         plt.vlines(x, 0, 5, color='gray', label='bin cells')
#     else:
#         plt.vlines(x, 0, 5, color='gray')
#     if x < 5:
#         plt.text(x+0.5, -0.5, x)

# for y in [0,1,2,3,4, 5]:
#     plt.hlines(y, 0, 5, color='gray')
#     if y < 5:
#         plt.text(-0.5, y+0.5, y)

# plt.hlines(0, 0, 5, color='black', linewidth=3, label='border')

# plt.scatter(Y[0,0], Y[0,1], s=20, marker='x', color='black', label='to estimate')

# colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
# for i in [0,1,2]:
    
#     Yx = Y[i,0]
#     Yy = Y[i,1]
    
#     plt.scatter(Yx, Yy, s=20, color=colors[i], marker='x')
    
# plt.scatter(X_star[:4,0]+0.5, X_star[:4,1]+0.5, marker='s', color='tab:blue')
# plt.scatter(X_star[4:,0]+0.5, X_star[4:,1]+0.5, marker='s', alpha=0.3, color='tab:blue')
# plt.scatter(0.1,6.45,marker='x', color='tab:blue')
# plt.vlines(0, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.vlines(3, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.text(0,5.5,'    exploring :\nfirst select all close enough centers\naccording to $x_0$')

# #%%
# fig, axs = plt.subplots(1,1)
# axs.set_aspect('equal', 'box')

# plt.axis('off')

# for x in [0,1,2,3,4, 5]:
#     if x == 0:
#         plt.vlines(x, 0, 5, color='gray', label='bin cells')
#     else:
#         plt.vlines(x, 0, 5, color='gray')
#     if x < 5:
#         plt.text(x+0.5, -0.5, x)

# for y in [0,1,2,3,4, 5]:
#     plt.hlines(y, 0, 5, color='gray')
#     if y < 5:
#         plt.text(-0.5, y+0.5, y)

# plt.hlines(0, 0, 5, color='black', linewidth=3, label='border')

# plt.scatter(Y[0,0], Y[0,1], s=20, marker='x', color='black', label='to estimate')

# colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
# for i in [0,1,2]:
    
#     Yx = Y[i,0]
#     Yy = Y[i,1]
    
#     plt.scatter(Yx, Yy, s=20, color=colors[i], marker='x')
    
# plt.scatter(X_star[:3,0]+0.5, X_star[:3,1]+0.5, marker='s', color='tab:blue')
# plt.scatter(X_star[3:,0]+0.5, X_star[3:,1]+0.5, marker='s', alpha=0.3, color='tab:blue')
# plt.vlines(0, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.vlines(3, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.hlines(0, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.hlines(2, 0,5,colors='tab:blue', ls='--', lw=3)
# plt.scatter(0.1,6.45,marker='x', color='tab:blue')
# plt.text(0,5.5,'    exploring :\nthen keep only close enough centers\naccording to $x_1$')

#%%
fig, axs = plt.subplots(1,1)
axs.set_aspect('equal', 'box')

plt.axis('off')

for x in [0,1,2,3,4, 5]:
    if x == 0:
        plt.vlines(x, 0, 5, color='gray', label='bin cells')
    else:
        plt.vlines(x, 0, 5, color='gray')
    if x < 5:
        plt.text(x+0.5, -0.5, x)

for y in [0,1,2,3,4, 5]:
    plt.hlines(y, 0, 5, color='gray')
    if y < 5:
        plt.text(-0.5, y+0.5, y)

# plt.hlines(0, 0, 5, color='black', linewidth=3, label='border')

plt.scatter(Y[0,0], Y[0,1], s=20, color='tab:orange', label='to estimate')

colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green'}
# for i in [0,1,2]:
    
#     Yx = Y[i,0]
#     Yy = Y[i,1]
    
#     plt.scatter(Yx, Yy, s=20, color=colors[i], marker='x')

for i, x in enumerate(X_star):
    plt.text(x[0]+0.4, x[1]+0.37, round(mu[i],2))

C = [1.5, 2.5]

plt.plot([C[0]-1.5, C[0] + 1.5, C[0] + 1.5, C[0] - 1.5, C[0] - 1.5],
          [C[1] - 1.5, C[1] - 1.5, C[1] + 1.5, C[1] + 1.5, C[1] - 1.5], color='tab:orange', ls='--', linewidth=3, label='kernel support')


plt.text(5.5, 3, '$\\tilde{p} = $', fontsize=20)
plt.text(6.5, 2.2, '$\\frac{2}{10} K(\\frac{\\sqrt{2}}{2} \\frac{2hs}{q})$', fontsize=20)
plt.text(6.1, 1.2, '$+ \\frac{3}{10} K(\\frac{2hs}{q})$', fontsize=20)
plt.text(6.1, 0.2, '$+ \\frac{2}{10} K(0)$', fontsize=20)

plt.arrow(2.7,3.3,6.5-2.7-0.1,2.2-3.3+0.1, shape='full', width=0.07, facecolor='gray', edgecolor='none', length_includes_head=True)

plt.arrow(2.7,2.3,6.1-2.7-0.1,1.2-2.3+0.1, shape='full', width=0.07, facecolor='gray', edgecolor='none', length_includes_head=True)

plt.arrow(1.7, 2.3, 6.1-1.7-0.1,0.2-2.3+0.3, shape='full', width=0.07, facecolor='gray', edgecolor='none', length_includes_head=True)

plt.legend(loc='lower left', bbox_to_anchor=(1,0.71))
plt.savefig('/home/frem/Work/LUCC/gisworknotes/Articles/LUCCProbabilitiesEstimation_EnvModSoft/figures/ekde_3.pdf', bbox_inches='tight')