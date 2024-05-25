#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from collections import Counter

#%% Importing dataset

pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/5/FM MP T1000 N8 sd10 pos.npy')
p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/5/FM MP T1000 N8 sd10 p.npy')
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/5/FM MP T1000 N8 sd10 ic.csv')

#%% System parameters

T = int(ic.loc[0][1])
dt = float(ic.loc[1][1])

if T < 1000:
    diff = 1000 - T
    T += diff
    steps = int(ic.loc[2][1]) + int(diff/dt)
    steps_truncated = int(ic.loc[15][1]) + diff
else:
    steps = int(ic.loc[2][1])
    steps_truncated = int(ic.loc[15][1])
    
N = int(ic.loc[6][1])
a = int(ic.loc[3][1])
sd = float(ic.loc[8][1])
system = (ic.loc[7][1])

# Mesghrid parameters
z = int(ic.loc[5][1])

x = np.linspace(-a/2, a/2, z)
y = np.linspace(-a/2, a/2, z)
X, Y = np.meshgrid(x,y)

truncation_point = int(ic.loc[16][1])
pos_truncated = pos[:,:,::truncation_point]
p_truncated = p[:,:,::truncation_point]

#%%
distances = np.zeros([N, N, steps_truncated])
indices = np.zeros([N, N, steps_truncated])

for n in range(0, steps_truncated):
    pos_temp = np.transpose(pos_truncated[:,:,n])
    nbrs_001 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos_temp)
    distances[:,:,n], indices[:,:,n] = nbrs_001.kneighbors(pos_temp)

    indices[:,:,n] = indices[:,:,n]*(distances[:,:,n] < 1.5)

#%%
counter = np.zeros([N,steps_truncated])

for n in range(0,steps_truncated):
    for i in range(0,N):
        for j in range(0,N):
            counter[i,n] += 1*(indices[i,j,n] != 0)

master_counter = np.zeros(steps_truncated+1)

for n in range(0,steps_truncated):
    master_counter[n] = sum(counter[:,n])
#%%
plt.cla()
plt.clf()
counts, bins = np.histogram(master_counter)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

#%%
t = np.linspace(0,T,steps_truncated+1)

plt.bar(t,master_counter)
plt.show()

#%%
def tuc(counter):
    tuc = []
    for n in range(0,steps_truncated):
        if counter[n] > 0:
            tuc.append(n)
    return tuc

tuc1 = tuc(counter[0,:])
tuc2 = tuc(counter[1,:])
tuc3 = tuc(counter[2,:])
tuc4 = tuc(counter[3,:])
tuc5 = tuc(counter[4,:])
tuc6 = tuc(counter[5,:])
tuc7 = tuc(counter[6,:])
tuc8 = tuc(counter[7,:])

#%%
def tof(tofp):
    tof = []
    
    for i in range(0,(np.shape(tofp)[0])-1):
        a = tofp[i]
        b = tofp[i+1]

        if b - a > 1:
            tof.append(b - a)
            
    if not tof:
        tof.append(0)
    return tof

tof_1 = tof(tuc1)
tof_2 = tof(tuc2)
tof_3 = tof(tuc3)
tof_4 = tof(tuc4)
tof_5 = tof(tuc5)
tof_6 = tof(tuc6)
tof_7 = tof(tuc7)
tof_8 = tof(tuc8)

total_tof = tof_1 + tof_2 + tof_3 + tof_4 + tof_5 + tof_6 + tof_7 + tof_8
#%% Stat Analysis

avg1 = np.average(tof_1)
avg2 = np.average(tof_2)
avg3 = np.average(tof_3)
avg4 = np.average(tof_4)
avg5 = np.average(tof_5)
avg6 = np.average(tof_6)
avg7 = np.average(tof_7)
avg8 = np.average(tof_8)

#%%
avg = np.average((avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8))
stdev = np.std((avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8))

#%% Saving
seed = 4

df = pd.DataFrame([['Average', avg], ['Standard deviation', stdev], 
                   ['A1', avg1], ['A2', avg2], ['A3', avg3], ['A4', avg4],
                   ['A5', avg5], ['A6', avg6], ['A7', avg7], ['A8', avg8]])

save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/Analysis/N8/TTC/'

saving_string = 'TTC SP' + str(N) + ' ' + system + 'M sd' + str(int(sd)) + ' seed' + str(seed)
df.to_csv(save_path + saving_string + ' stats.csv')
np.save(save_path + saving_string + ' total tof', total_tof)

#%%

fig, ax1 = plt.subplots(figsize=(7,6))

for n in range(0, steps+1, 400):
    plt.cla()
    for i in range(N):
        ax1.set_aspect(1)
        circle = plt.Circle((pos[0,i,n], pos[1,i,n]), 0.5, edgecolor = 'r', facecolor = 'Gainsboro')
        ax1.add_artist(circle)
        
        plt.annotate(str(i), (pos[:,i,n]), fontsize = 16)
        
        ax1.arrow(pos[0,i,n], pos[1,i,n], p[0,i,n]/2, p[1,i,n]/2, head_width=0.1, head_length=0.1, fc='k', ec='k')
        ax1.arrow(pos[0,i,n], pos[1,i,n], -p[0,i,n]/2, -p[1,i,n]/2, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax1.axis([-a/2, a/2, -a/2, a/2])
    ax1.tick_params(top = True, labeltop = True, right = True, labelright = True)
    
    plt.title('%3d' %(n*dt))

    plt.pause(0.5)
    # plt.savefig(str(n) + ' SP'  +str(SP) + '.png', bbox_inches = 'tight')