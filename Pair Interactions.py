#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,-1]
p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic p.npy')[:,:,-1]

#%% pair interactions
N = pos.shape[1]
bins = np.array([0,1,2,3])
bin_counts = np.zeros(len(bins) - 1)

for i in range(0, N):
    origin = np.abs(pos[:, 0]) * np.abs(p[:, 0])
    origin = origin[:, np.newaxis]* np.ones([624])
    p_origin = p[:, 0]
    p_origin = p_origin[:, np.newaxis] * np.ones([624])
    r = pos[:, 1:] - origin
    rdotp = np.dot(np.transpose(r), p_origin)
    rdotp = rdotp[:, np.newaxis]  
    rdotpp = r * rdotp
    rabs = r - rdotpp
    r_scalar = np.sqrt(rabs[0, :]**2 + rabs[1, :]**2)
    r_scalar = r_scalar * (r_scalar < 3)
    bin_indices = np.digitize(r_scalar, bins)
    for j in range(1, len(bins)):
        bin_counts[j-1] += np.sum(bin_indices == j)
    pos = np.roll(pos, 1, axis=1)
    p = np.roll(p, 1, axis=1)

# Optionally, normalize bin counts by N
bin_counts = bin_counts/N

# %%
