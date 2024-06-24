#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% importing files
pos_f01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/FM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,::20]
pos_01_1 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic pos.npy')[:,:,::20]
pos_01_2 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic continued pos.npy')[:,:,::int(1/0.05)]
pos_01 = np.concatenate((pos_01_1, pos_01_2), axis = 2)
pos_001_1 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,::20]
pos_001_2 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic continued pos.npy')[:,:,::int(1/0.1)]
pos_001_3 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued pos.npy')[:,:,::int(1/0.1)]
pos_001 = np.concatenate((pos_001_1, pos_001_2, pos_001_3), axis = 2)
pos_0001_1 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.01 isotropic pos.npy')[:,:,::20]
pos_0001_2 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.01 isotropic continued pos.npy')[:,:,::int(1/0.5)]
pos_0001_3 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued pos.npy')[:,:,::int(1/0.5)]
pos_0001_4 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 2 pos.npy')[:,:,::int(1/0.5)]
pos_0001 = np.concatenate((pos_0001_1, pos_0001_2, pos_0001_3, pos_0001_4), axis = 2)
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic ic.csv')

N = int(ic.loc[4][1])

#%%
def calculate_msd(positions):
    _, N, t = positions.shape
    msd = np.zeros(t)
    origin = np.zeros([2,N])
    
    for time in range(t):
        displacements = positions[:, :, time] - origin
        scalar_displacements = np.sqrt(displacements[0]**2 + displacements[1]**2)
        msd[time] = np.mean(scalar_displacements)
    
    return msd

msd_f01 = calculate_msd(pos_f01)
msd_01 = calculate_msd(pos_01)
msd_001 = calculate_msd(pos_001)
msd_0001 = calculate_msd(pos_0001)

# %%
t1 = np.linspace(0, 5001, 5000)
t10 = np.linspace(0, 10001, 10000)
t2 = np.linspace(0, 20001, 20000)
t3 = np.linspace(0, 30001, 30000)
plt.figure(figsize=(10, 6))
plt.plot(t1[1:], msd_f01[1:-1], label = '$FM, \ell = 0.1$')
plt.plot(t10[1:], msd_01[1:-2], label = '$CM, \ell H = 0.1$')
plt.plot(t2[1:],msd_001[1:-3], label = '$CM, \ell H = 0.01$')
plt.plot(t3[1:],msd_0001[1:-4], label = '$CM, \ell H = 0.001$')
plt.xlabel('Time')
plt.ylabel('Mean Squared Displacement')
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.semilogx(t1[1:], msd_f01[1:-1], label = '$FM, \ell = 0.1$')
plt.semilogx(t2[1:], msd_01[1:-2], label = '$CM, \ell H = 0.1$')
plt.semilogx(t2[1:], msd_001[1:-2], label = '$CM, \ell H = 0.01$')
plt.semilogx(t3[1:], msd_0001[1:-4], label = '$CM, \ell H = 0.001$')
plt.xlabel('Time')
plt.ylabel('Mean Squared Displacement')
plt.legend()
plt.show()

#%%
plt.figure(figsize=(10, 6))
plt.loglog(t1[1:], msd_f01[1:-1], label = '$FM, \ell = 0.1$')
plt.loglog(t2[1:], msd_01[1:-2], label = '$CM, \ell H = 0.1$')
plt.loglog(t2[1:], msd_001[1:-2], label = '$CM, \ell H = 0.01$')
plt.loglog(t3[1:], msd_0001[1:-4], label = '$CM, \ell H = 0.001$')
plt.xlabel('Time')
plt.ylabel('Mean Squared Displacement')
plt.legend()
plt.show()
# %%
