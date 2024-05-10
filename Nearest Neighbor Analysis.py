#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter

#%%
pos_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,30000]
pos_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic pos.npy')[:,:,30000]
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic ic.csv')

#%%
N = int(ic.loc[4][1])
# T = int(ic.loc[0][1]) 
# steps = int(ic.loc[2][1])

#%%
distances_001 = np.zeros([N, N])
indices_001 = np.zeros([N, N])
distances_01 = np.zeros([N, N])
indices_01 = np.zeros([N, N])

# for n in range(steps-3,steps):
pos001 = np.transpose(pos_001[:,:])
pos01 = np.transpose(pos_01[:,:])
nbrs_001 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos001)
distances_001[:,:], indices_001[:,:] = nbrs_001.kneighbors(pos001)

nbrs_01 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos01)
distances_01[:,:], indices_01[:,:] = nbrs_01.kneighbors(pos01)

indices_001[:,:] = indices_001[:,:]*(distances_001[:,:] < 1.5)
indices_01[:,:] = indices_01[:,:]*(distances_01[:,:] < 1.5)

#%%
counter_01 = np.zeros(N)
counter_001 = np.zeros(N)

# for n in range(0,steps):
for i in range(0,N):
    for j in range(0,N):
        counter_01[i] += 1*(indices_01[i,j] != 0)
        counter_001[i] += 1*(indices_001[i,j] != 0)

# %% probability

def probability(counts):
    a = Counter(counts)
    p1 = a[1]/N
    p2 = a[2]/N
    p3 = a[3]/N
    p4 = a[4]/N
    p5 = a[5]/N
    
    prob = np.array([p1, p2, p3, p4, p5])
    
    return prob

probability_01 = probability(counter_01)
probability_001 = probability(counter_001)

f = 14

plt.cla()
plt.clf()
plt.bar([1,2,3,4,5], probability_001, hatch ="/", label = '$\ell H = 0.01$', color = 'black')
plt.bar([1,2,3,4,5], probability_01, edgecolor = 'k', hatch = "//", label = '$\ell H = 0.1$', color = 'salmon', alpha = 0.5)
plt.legend(fontsize = f)
plt.xlabel('Number of neighbors, N', fontsize = f)
plt.ylabel('$p(N)$', fontsize = f)
plt.savefig('Nearest Neighbor Analysis, N625.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# %%
