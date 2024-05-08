  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 18:22:02 2023

@author: fizzausmani
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interp

#%% Source dipoles

pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/CM MP test2 T500 N3 sd0.1 pos.npy')
p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/CM MP test2 T500 N3 sd0.1 p.npy')
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/CM MP test2 T500 N3 sd0.1 ic.csv')

#%% Meshgrid
ux = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/CM MP test2 T500 N3 sd0.1 ux.npy')
uy = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/CM MP test2 T500 N3 sd0.1 uy.npy')
# c = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/seed0/Mixing/AD C+D RK4 BR_1 T1000 dt0.05 SP4 FM sd1000 conc.npy')

#%% System parameters

T = int(ic.loc[0][1])
dt = float(ic.loc[1][1])

if T < 500:
    diff = 500 - T
    T += diff
    steps = int(ic.loc[2][1]) + int(diff/dt)
    steps_truncated = int(ic.loc[15][1]) + diff
else:
    steps = int(ic.loc[2][1])
    steps_truncated = int(ic.loc[15][1])
    
N = int(ic.loc[6][1])
a = int(ic.loc[3][1])

#%% Mesghrid parameters
z = int(ic.loc[5][1])
x = np.linspace(-a/2, a/2, z)
y = np.linspace(-a/2, a/2, z)
X, Y = np.meshgrid(x,y)

xx = x
yy = y

truncation_point = int(ic.loc[16][1])
pos_t = pos[:,:,::truncation_point]
p_t = p[:,:,::truncation_point]

# %% Centre box view

fig, ax1 = plt.subplots(figsize=(7,6))

for n in range(0, steps+1, 20):
    
    plt.cla()
    for i in range(N):
        ax1.set_aspect(1)
        circle = plt.Circle((pos[0,i,n], pos[1,i,n]), 0.5, edgecolor = 'r',\
                            facecolor = 'Gainsboro')
        ax1.add_artist(circle)
        
        ax1.arrow(pos[0,i,n], pos[1,i,n], p[0,i,n]/2, p[1,i,n]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')
        ax1.arrow(pos[0,i,n], pos[1,i,n], -p[0,i,n]/2, -p[1,i,n]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')

    ax1.axis([-a/2, a/2, -a/2, a/2])
    ax1.tick_params(top = True, labeltop = True, right = True, labelright = True)
    
    plt.title('%3d' %(n*dt))

    # plt.pause(0.01)
    plt.savefig(str(n) + ' SP'  +str(N) + '.jpeg', bbox_inches = 'tight', dpi = 300)

#%% Velocity

# plot = 'Q'
plot = 'S'

figure, axes = plt.subplots(figsize=(10,8))

# n = 0
# if n == 0:
for n in range(0, T):
    plt.cla()
    for i in range(N):
        umag = np.sqrt(ux[:,:,n]**2+uy[:,:,n]**2)
        
        axes.set_aspect(1)
        
        circle = plt.Circle((pos_t[0,i,n], pos_t[1,i,n]), 0.5,
                            edgecolor = 'k', facecolor = 'Gainsboro', zorder = 10)
        
        axes.arrow(pos_t[0,i,n], pos_t[1,i,n], p_t[0,i,n]/2,\
                   p_t[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
        axes.arrow(pos_t[0,i,n], pos_t[1,i,n], -p_t[0,i,n]/2,\
                   -p_t[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
        
        s = 5 #spacer

        if plot == 'Q':
            CS = axes.contourf(x,y,umag, 200, cmap = 'twilight_r', zorder  = 1)
            # axes.quiver(X[::s,::s], Y[::s,::s], ux[::s,::s,n], uy[::s,::s,n],\
            #             pivot = 'mid', scale = 2)
            
        elif plot == 'S':
            CS = axes.contourf(x,y,umag, 200, cmap = 'twilight_r', zorder  = 1)
            plt.streamplot(X[::s,::s], Y[::s,::s], ux[::s,::s,n], uy[::s,::s,n],\
                           color = 'k')
            
        axes.add_artist(circle)
        
    if n == 0:
        cbar = figure.colorbar(CS, pad = 0.08)
    else:
        # cbar.update_normal(CS)
        cbar.update_ticks()
        
    cbar.ax.set_ylabel('Velocity')
    
    axes.axis([-a/2, a/2, -a/2, a/2])
    axes.tick_params(top = True, labeltop = True, right = True, labelright = True)
    
    plt.title('%3d' %(n))
    # plt.pause(0.001)
    plt.savefig(str(n) + '.png', bbox_inches = 'tight', dpi = 300)
    
    progress = (n)/T*100
    if progress in np.linspace(0,100,21):
        print('Progress is ' + str(progress) + '%')


#%% Concentration

figure, axes = plt.subplots(figsize=(10,8))

for n in range(0,T):
    plt.cla()
    for i in range(0,N):
        m = n
        axes.set_aspect(1)
        
        circle = plt.Circle((pos_t[0,i,m], pos_t[1,i,m]), 0.5,\
                            edgecolor = 'r', facecolor = 'Gainsboro', zorder = 10)
        
        circle = plt.Circle((pos_t[0,i,n], pos_t[1,i,n]), 0.5,
                            edgecolor = 'k', facecolor = 'Gainsboro', zorder = 10)
        axes.arrow(pos_t[0,i,n], pos_t[1,i,n], p_t[0,i,n]/2,\
                   p_t[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
        axes.arrow(pos_t[0,i,n], pos_t[1,i,n], -p_t[0,i,n]/2,\
                   -p_t[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
            
        CS = axes.contourf(x, y, c[:,:,n], 100, cmap = 'twilight', zorder  = 1)
        
        axes.add_artist(circle)
        
    if n == 0:
        cbar = figure.colorbar(CS, pad = 0.01,ticks=[-1, -0.5, 0, 0.5, 1])
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1], labels = [-1, -0.5, 0, 0.5, 1],\
                       fontsize = 12)
        # cbar.tick_params(labelsize = 10)
    else:
        # cbar.update_normal(CS)
        cbar.update_ticks()
    
    axes.axis([-a/2, a/2, -a/2, a/2])
    axes.tick_params(left = False, labelleft = False, bottom = False,\
                     labelbottom = False, top = False, labeltop = False,\
                         right = False, labelright = False)
    
    plt.title((n))
    # plt.pause(0.01)
    
    plt.savefig(str(n) + '.png', bbox_inches = 'tight')

    progress = (n)/T*100
    if progress in np.linspace(0,100,21):
        print('Progress is ' + str(progress) + '%')
