#%%
import numpy as np
import matplotlib.pyplot as plt

pos_new = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic tester pos.npy')
pos_old = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic continued pos.npy')

diff = pos_new - pos_old
np.amax(diff)
# %%
