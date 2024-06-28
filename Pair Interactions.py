#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic pos.npy')[:,:,-1]
p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic p.npy')[:,:,-1]

#%% pair interactions
N = np.shape(pos[1])
rfx=[]
rfy=[]

for i in range(0, N):
    origin = (pos[:, 0]) 
    origin = origin[:, np.newaxis]* np.ones([N])
    p_origin = (p[:, 0])
    p_origin = p_origin[:, np.newaxis] * np.ones([N])
    r = pos - origin
    x = np.dot(np.transpose(r), p_origin)
    x = x[:, 0]  
    rdotpp = p_origin*x
    y_vec = r - rdotpp
    # check = np.dot(np.transpose(rabs), p_origin)   
    y = np.sqrt(y_vec[0, :]**2 + y_vec[1, :]**2)
    rfx.append(np.abs(x))
    rfy.append(y)
    pos = np.roll(pos, 1, axis=1)
    p = np.roll(p, 1, axis=1)

rfx = np.array(rfx)
rfy = np.array(rfy)

rfx_reshaped = np.ravel(rfx)
rfy_reshaped = np.ravel(rfy)
plt.scatter(rfy_reshaped, rfx_reshaped, s = 0.5, alpha = 0.5)
plt.xlim(0,4)
plt.ylim(0,4)
