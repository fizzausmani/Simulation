#%%
import numpy as np
import matplotlib.pyplot as plt

pos_new = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T2000 N625 lattice sd1 isotropic testing pos.npy')[:,:,-1]
pos_old = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic pos.npy')[:,:,int(2000/0.05)]

diff = pos_new - pos_old
np.amax(diff)
# %%
