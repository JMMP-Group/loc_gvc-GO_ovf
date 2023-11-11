#!/usr/bin/env python

#     |---------------------------------------------------------------|
#     | This module computes the initial density stratification of    |
#     | the ocean and of the cold water mass in the Denmark Strait    |
#     | for the idealised overflow experiment.                        |
#     | The method is similar to what was done in                     |
#     | Riemenschneider & Legg 2007, doi:10.1016/j.ocemod.2007.01.003 |
#     |                                                               |
#     | Author: Diego Bruciaferri                                     |
#     | Date and place: 03-12-2021, Met Office, UK                    |
#     |---------------------------------------------------------------|

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

nx   = 20
Dist = np.linspace(0.0,1.0,nx)
dist = 1. - Dist

alpha = [0.25, 0.5, 1., 2., 4.]
#color = ['violet', 'red', 'black', 'magenta','blue','deepskyblue','green']

fig  = plt.figure(figsize=(6,10))
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax1  = fig.add_subplot(spec[:1])

for a in range(len(alpha)):

    w = 0.5*(np.tanh(alpha[a]*(2*Dist/(Dist + dist)-1))/np.tanh(alpha[a])+1)

    #ax1.plot(range(nx), w, color[a], linewidth=4)
    Lab = r'$\mu = $'+str(alpha[a])
    ax1.plot(Dist, w, linewidth=2.5, label=Lab)

ax1.set_ylabel('$W_p$', fontsize='25',color="black")
ax1.set_xlabel('$D_p \;/ \; \max{(D_p)}$', fontsize='25',color="black")
ax1.tick_params(axis='y', labelsize=25)
ax1.tick_params(axis='x', labelsize=25, which='major', width=1.50, length=10, labelcolor="black")

plt.rc('legend', **{'fontsize':20})
ax1.legend(loc='best', frameon=False)
fig_name = 'mu_sens.jpg'
plt.savefig(fig_name, bbox_inches="tight")
print("done")
plt.close()

