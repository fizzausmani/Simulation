#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:08:06 2023

@author: fizzausmani
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp
import math

#%% Loading meshgrid velocity and system condition files
OS = 'M'

if OS == 'M':
    ux = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd1 ux.npy')
    uy = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd1 uy.npy')
    info = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd1 ic.csv')
    save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/Mixing/'

elif OS == 'W':
    ux = np.load(r'C:\Users\fusmani\Box\Research\Python\Mixing\No Brownian\Data\10x10\FM T1000 SP6 sd1000.0 60x60 ux.npy')
    uy = np.load(r'C:\Users\fusmani\Box\Research\Python\Mixing\No Brownian\Data\10x10\FM T1000 SP6 sd1000.0 60x60 uy.npy')
    info = pd.read_csv(r'C:\Users\fusmani\Box\Research\Python\Mixing\No Brownian\Data\10x10\FM T1000 SP6 sd1000.0 60x60 ic.csv')
    save_path = 'C:\\Users\\fusmani\\Box\\Research\\Python\\Full Results\\SP6\\Mixing\\'

#%% Meshgrid & time parameters
n = 0 # number of continued simulations
a = int(info.loc[3][1]) # length of box
z = int(info.loc[5][1]) # length of x,y array

x = np.linspace(-a, a, z)
dx = x[1] - x[0]
y = np.linspace(-a, a, z)
X, Y = np.meshgrid(x,y)

SP = int(info.loc[6][1])
system = info.loc[7][1]
sd = info.loc[8][1]

X, Y = np.meshgrid(x,y)

T = int(info.loc[0][1])
dt = float(info.loc[1][1])

if T < 1000:
    diff = 1000 - T
    T += diff
    steps = int(info.loc[2][1]) + int(diff/dt)
    steps_truncated = int(info.loc[15][1]) + diff
else:
    steps = int(info.loc[2][1])
    steps_truncated = int(info.loc[15][1])
    
truncation_point = int(info.loc[16][1])
t = np.linspace(0, T, steps_truncated)

dt = 0.05
steps = int(T/dt)

#%% Concentration initialization
conc = np.zeros([z,z])
k1 = np.zeros([z,z])
conc_full = np.zeros([z, z, steps_truncated+1])

for i in range(0,z):
    for j in range(0,z):
        conc[i,j] = (np.sin((2*np.pi/a)*y[i]))

# Below part commented out if running code with actual velocities
# for n in range(0,steps_truncated+1):
#     b = 2
#     omega = 2
#     for i in range(0,zz):
#         for j in range(0,zz):
#             ux[i,j,n] = np.sin((2*np.pi/a)*XX[i,j]+b*np.sin(omega*t[n]))*np.cos((2*np.pi/a)*YY[i,j])
#             uy[i,j,n] = np.sin((2*np.pi/a)*YY[i,j])*-np.cos((2*np.pi/a)*XX[i,j])
        # uy[i,j,:] = 1

#%% Spatial differences - 2D
def order1_2D(conc_i,n):
    N = math.floor(n*dt)
    
    # central cells
    k1[1:-1,1:-1] = 1/dx*((uy[1:-1,1:-1,N] > 0)*(uy[1:-1,1:-1,N])*(conc_i[1:-1,1:-1] - conc_i[0:-2,1:-1])
                       + (uy[1:-1,1:-1,N] < 0)*(uy[1:-1,1:-1,N])*(conc_i[2:,1:-1] - conc_i[1:-1,1:-1])
                       + (ux[1:-1,1:-1,N] > 0)*(ux[1:-1,1:-1,N]*(conc_i[1:-1,1:-1] - conc_i[1:-1,0:-2]))
                       + (ux[1:-1,1:-1,N] < 0)*(ux[1:-1,1:-1,N]*(conc_i[1:-1,2:] - conc_i[1:-1,1:-1])))
    
    # columns and rows on the edges
    # left column excluding corners
    k1[1:-1,0] = 1/dx*((uy[1:-1,0,N] > 0)*(uy[1:-1,0,N])*(conc_i[1:-1,0] - conc_i[0:-2,0])
                       + (uy[1:-1,0,N] < 0)*(uy[1:-1,0,N])*(conc_i[2:,0] - conc_i[1:-1,0])
                       + (ux[1:-1,0,N] > 0)*(ux[1:-1,0,N]*(conc_i[1:-1,0] - conc_i[1:-1,-2]))
                       + (ux[1:-1,0,N] < 0)*(ux[1:-1,0,N]*(conc_i[1:-1,1] - conc_i[1:-1,0])))

    # right column excluding corners
    k1[1:-1,-1] = 1/dx*((uy[1:-1,-1,N] > 0)*(uy[1:-1,-1,N])*(conc_i[1:-1,-1] - conc_i[0:-2,-1])
                       + (uy[1:-1,-1,N] < 0)*(uy[1:-1,-1,N])*(conc_i[2:,-1] - conc_i[1:-1,-1])
                       + (ux[1:-1,-1,N] > 0)*(ux[1:-1,-1,N]*(conc_i[1:-1,-1] - conc_i[1:-1,-2]))
                       + (ux[1:-1,-1,N] < 0)*(ux[1:-1,-1,N]*(conc_i[1:-1,1] - conc_i[1:-1,-1])))    
    
    # top row excluding corners
    k1[0,1:-1] = 1/dx*((uy[0,1:-1,N] > 0)*(uy[0,1:-1,N])*(conc_i[0,1:-1] - conc_i[-2,1:-1])
                       + (uy[0,1:-1,N] < 0)*(uy[0,1:-1,N])*(conc_i[1,1:-1] - conc_i[0,1:-1])
                       + (ux[0,1:-1,N] > 0)*(ux[0,1:-1,N]*(conc_i[0,1:-1] - conc_i[0,0:-2]))
                       + (ux[0,1:-1,N] < 0)*(ux[0,1:-1,N]*(conc_i[0,2:] - conc_i[0,1:-1])))
    
    # bottom row excluding corners
    k1[-1,1:-1] = 1/dx*((uy[-1,1:-1,N] > 0)*(uy[-1,1:-1,N])*(conc_i[-1,1:-1] - conc_i[-2,1:-1])
                       + (uy[-1,1:-1,N] < 0)*(uy[-1,1:-1,N])*(conc_i[1,1:-1] - conc_i[-1,1:-1])
                       + (ux[-1,1:-1,N] > 0)*(ux[-1,1:-1,N]*(conc_i[-1,1:-1] - conc_i[-1,0:-2]))
                       + (ux[-1,1:-1,N] < 0)*(ux[-1,1:-1,N]*(conc_i[-1,2:] - conc_i[-1,1:-1])))
    
    # corners
    # top-left corner
    k1[0,0] = 1/dx*((uy[0,0,N] > 0)*(uy[0,0,N])*(conc_i[0,0] - conc_i[-2,0])
                       + (uy[0,0,N] < 0)*(uy[0,0,N])*(conc_i[1,0] - conc_i[0,0])
                       + (ux[0,0,N] > 0)*(ux[0,0,N]*(conc_i[0,0] - conc_i[0,-2]))
                       + (ux[0,0,N] < 0)*(ux[0,0,N]*(conc_i[0,1] - conc_i[0,0])))
    
    #bottom-left corner
    k1[-1,0] = 1/dx*((uy[-1,0,N] > 0)*(uy[-1,0,N])*(conc_i[-1,0] - conc_i[-2,0])
                       + (uy[-1,0,N] < 0)*(uy[-1,0,N])*(conc_i[1,0] - conc_i[-1,0])
                       + (ux[-1,0,N] > 0)*(ux[-1,0,N]*(conc_i[-1,0] - conc_i[-1,-2]))
                       + (ux[-1,0,N] < 0)*(ux[-1,0,N]*(conc_i[-1,1] - conc_i[-1,0])))
    
    # top-right corner
    k1[0,-1] = 1/dx*((uy[0,-1,N] > 0)*(uy[0,-1,N])*(conc_i[0,-1] - conc_i[-2,-1])
                       + (uy[0,-1,N] < 0)*(uy[0,-1,N])*(conc_i[1,-1] - conc_i[0,-1])
                       + (ux[0,-1,N] > 0)*(ux[0,-1,N]*(conc_i[0,-1] - conc_i[0,-2]))
                       + (ux[0,-1,N] < 0)*(ux[0,-1,N]*(conc_i[0,1] - conc_i[0,-1])))
    
    #bottom-right corner
    k1[-1,-1] = 1/dx*((uy[-1,-1,N] > 0)*(uy[-1,-1,N])*(conc_i[-1,-1] - conc_i[-2,-1])
                       + (uy[-1,-1,N] < 0)*(uy[-1,-1,N])*(conc_i[1,-1] - conc_i[-1,-1])
                       + (ux[-1,-1,N] > 0)*(ux[-1,-1,N]*(conc_i[-1,-1] - conc_i[-1,-2]))
                       + (ux[-1,-1,N] < 0)*(ux[-1,-1,N]*(conc_i[-1,1] - conc_i[-1,-1])))
    
    return k1

#%% Concentration calculation

for t in range(0, steps):
    k1 = -order1_2D(conc, t)
    temp_conc = (k1/2)*dt + conc
    k2 = -order1_2D(temp_conc, t)
    temp_conc2 = (k2/2)*dt + conc
    k3 = -order1_2D(temp_conc2, t)
    temp_conc = k3*dt + conc
    k6 = -order1_2D(temp_conc, t)
    
    conc += dt*(k1+2*k2+2*k3+k6)/6
    
    # conc += dt*k1

    progress = (t*dt)/T*100
    if progress in np.linspace(0,100,21):
        print('Progress is ' + str(progress) + '%')
    
    t_save = np.arange(0, steps, int(1/dt))
    
    if t in t_save:
        conc_full[:,:,math.floor(t*dt)] = conc[:,:]
    # conc_full[:,:,t] = conc[:,:]
        
#%% CFL max
CFLx_max = np.zeros(steps_truncated+1)
CFLy_max = np.zeros(steps_truncated+1)

for n in range(0, steps_truncated+1):
    CFLx_max[n] = np.amax(abs((ux[:,:,n]*dt)/dx)[:,:])
    CFLy_max[n] = np.amax(abs((uy[:,:,n]*dt)/dx)[:,:])
    
CFLx = max(CFLx_max)
CFLy = max(CFLy_max)

#%% Saving arrays
df = pd.DataFrame([['T', T], ['dt', dt],
                   ['CFLx', CFLx], ['CFLy', CFLy]])

Pe = int(info.loc[12][1])

if Pe <= 100:
    saving_string = 'AD RK4 BR_' + str(Pe) + ' T' + str(T) + ' dt' + str(dt) + ' SP' + str(SP) + ' ' + system +  'M sd' + str(sd)
else:
    saving_string = 'AD RK4 T' + str(T) + ' dt' + str(dt) + ' SP' + str(SP) + ' ' + system +  'M sd' + str(sd)

np.save(save_path + saving_string + ' conc', conc_full)
df.to_csv(save_path + saving_string + ' dt and CFL data.csv', index = False) 

#%% Plotting
if OS == 'M':
    pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd1 pos.npy')
    p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd1 p.npy')

elif OS == 'W':    
    pos_sources = np.load(r'C:\Users\fusmani\Box\Research\Python\Mixing\No Brownian\Data\20x20\CM T1000 SP6 sd1 dipole pos.npy') # positions
    ori = np.load(r'C:\Users\fusmani\Box\Research\Python\Mixing\No Brownian\Data\20x20\CM T1000 SP6 sd1 ori.npy') # orientations

pos_truncated = pos[:,:,::truncation_point]
p_truncated = p[:,:,::truncation_point]

#%%
figure, axes = plt.subplots(figsize=(10,8))

for n in range(0, T):
    plt.cla()
    for i in range(0,SP):
        m = n
        axes.set_aspect(1)
        
        circle = plt.Circle((pos_truncated[0,i,n], pos_truncated[1,i,n]), 0.6, edgecolor = 'r', facecolor = 'Gainsboro', zorder = 10)
        
        axes.arrow(pos_truncated[0,i,n], pos_truncated[1,i,n], p_truncated[0,i,n]/2, p_truncated[1,i,n]/2, head_width=0.1, head_length=0.1, fc='k', ec='k', zorder = 11)
        axes.arrow(pos_truncated[0,i,n], pos_truncated[1,i,n], -p_truncated[0,i,n]/2, -p_truncated[1,i,n]/2, head_width=0.1, head_length=0.1, fc='k', ec='k', zorder = 11)

        CS = axes.contourf(x, y, conc_full[:,:,n], cmap = 'twilight', zorder  = 1, levels = 200)

        axes.add_artist(circle)
        
    if n == 0:
        cbar = figure.colorbar(CS, pad = 0.01,ticks=[-1, -0.6, 0, 0.6, 1])
        cbar.set_ticks([-1, -0.6, 0, 0.6, 1], labels = [-1, -0.6, 0, 0.6, 1], fontsize = 12)
    else:
        cbar.update_ticks()
        
    axes.axis([-a/2, a/2, -a/2, a/2])
    axes.tick_params(left = False, labelleft = False, bottom = False, labelbottom = False, top = False, labeltop = False, right = False, labelright = False)
    
    plt.title((n))
    # plt.pause(0.01)
    # 
    plt.savefig(str(n) + ' mix.png', bbox_inches = 'tight')

    progress = (n)/T*100
    if progress in np.linspace(0,100,21):
        print('Progress is ' + str(progress) + '%')


#%% Total concentration and maximum of concentration calculation
total_conc = np.zeros(steps_truncated+1)
cmax = np.zeros(steps_truncated+1)

for n in range(0, steps_truncated+1):
    total_conc[n] = np.trapz(np.trapz(conc_full[:,:,n],x),x)
    cmax[n] = np.amax(conc_full[:,:,n])

f = 16
t = np.linspace(0, T, steps_truncated+1)

plt.plot(t, total_conc)
plt.xlabel('t', fontsize = f)
plt.ylabel('$C_{total}$', fontsize = f)
plt.title('$C_{total}$ RK6 dt ' + str(dt) + ', First Order Upwind', pad = 20, fontsize = f)
plt.show()