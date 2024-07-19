#%%
import numpy as np
import matplotlib.pyplot as plt

pos_new = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T4000 N625 lattice sd0.01 isotropic testing 2 pos.npy')[:,:,-1]
pos_old = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.01 isotropic pos.npy')[:,:,-1]

diff = pos_new - pos_old
np.amax(diff)
# %%
