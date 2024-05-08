#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from time import time

#%% Importing positions and velocities

pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd0.1 pos.npy')
ux = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd0.1 ux.npy')
uy = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd0.1 uy.npy')
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/CM MP T1000 N8 sd0.1 ic.csv')
save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/10x10/0/'

#%% System parameters
T = int(ic.loc[0][1])
dt = float(ic.loc[1][1])

N = int(ic.loc[6][1])
a = int(ic.loc[3][1])
z = int(ic.loc[5][1])
x = np.linspace(-a/2, a/2, z)
y = np.linspace(-a/2, a/2, z)
X,Y = np.meshgrid(x, y)
dx = x[1] - x[0]

system = ic.loc[7][1]
sd = float(ic.loc[8][1])

steps_truncated = int(ic.loc[15][1])

L = 0.02
Us = 5
beta = 20
D = 1

truncation_point = int(ic.loc[16][1])
pos_truncated = pos[:,:,::truncation_point]

# %% Velocity field extraction
grid_pos = np.array([np.ravel(X),np.ravel(Y)])

x_g = grid_pos[0,:]
x_g = x_g[:,np.newaxis]*np.ones([N*2*9])
y_g = grid_pos[1,:]
y_g = y_g[:,np.newaxis]*np.ones([N*2*9])

ux_h = np.zeros([z, z, steps_truncated+1])
uy_h = np.zeros([z, z, steps_truncated+1])


def mi(main):
    mi = np.zeros([2,N*8])
    mi_E = np.zeros([2,N])
    mi_W = np.zeros([2,N])
    mi_N = np.zeros([2,N])
    mi_S = np.zeros([2,N])
    mi_NE = np.zeros([2,N])
    mi_NW = np.zeros([2,N])
    mi_SE = np.zeros([2,N])
    mi_SW = np.zeros([2,N])
    for i in range(N):
        mi_E[:,i] = main[:,i] + np.array([a,0])
        mi_W[:,i] = main[:,i] + np.array([-a,0])
        mi_N[:,i] = main[:,i] + np.array([0,a])
        mi_S[:,i] = main[:,i] + np.array([0,-a])
        mi_NE[:,i] = main[:,i] + np.array([a,a])
        mi_NW[:,i] = main[:,i] + np.array([-a,a])
        mi_SE[:,i] = main[:,i] + np.array([a,-a])
        mi_SW[:,i] = main[:,i] + np.array([-a,-a])
    mi = np.concatenate((mi_E, mi_W, mi_N, mi_S, mi_NE, mi_NW, mi_SE, mi_SW), axis = 1)
    return mi

def u_extraction(ux, uy, pos):
    ux = np.ravel(ux)
    uy = np.ravel(uy)
    
    images = mi(pos[:,:])
    x_pos = np.concatenate((pos[0,:],images[0,:]), axis = 0)
    # x_pos = pos[0,:]
    x_center = np.repeat(x_pos,2)
    
    y_pos = np.concatenate((pos[1,:],images[1,:]), axis = 0)
    # y_pos = pos[1,:]
    y_center = np.repeat(y_pos,2)

    rmag_c = ((x_g-np.transpose(x_center))**2+(y_g-np.transpose(y_center))**2)**0.5
    selfcheck = (rmag_c>0.5)

    Ust_x = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))*((x_g-np.transpose(x_center))/rmag_c))
    Ust_y = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))*((y_g-np.transpose(y_center))/rmag_c))    
    Ust_x = Ust_x*selfcheck
    Ust_y = Ust_y*selfcheck
    Ust_x = np.sum(Ust_x, axis = 1)
    Ust_y = np.sum(Ust_y, axis = 1)

    ux -= Ust_x
    uy -= Ust_y

    ux_h = np.reshape(ux, [z,z])
    uy_h = np.reshape(uy, [z,z])

    return ux_h, uy_h


start_time=time()

for n in range(0, T):
    ux_h[:,:,n], uy_h[:,:,n] = u_extraction(ux[:,:,n], uy[:,:,n], pos_truncated[:,:,n])

end_time = time()
elapsed_time = end_time - start_time

#%%
saving_string = system + 'M MP T' + str(T) + ' N' + str(N) + ' sd' + str(sd)

np.save(save_path + saving_string + ' ux_h', ux_h)
np.save(save_path + saving_string + ' uy_h', uy_h)

# %% Plotting

plot = 'Q'
# plot = 'S'

figure, axes = plt.subplots(figsize=(10,8))

# vmin = 0
# vmax = 1
levels = np.linspace(0.0, 0.2, 7)

# n = 8
# if n == 8:
for n in range(0, 10):
    plt.cla()
    for i in range(N):
        umag = np.sqrt(ux_new[:,:,n]**2 + uy_new[:,:,n]**2)
        
        axes.set_aspect(1)
        
        circle = plt.Circle((pos_truncated[0,i,n], pos_truncated[1,i,n]), 0.5,
                            edgecolor = 'k', facecolor = 'Gainsboro', zorder = 10)
        
        axes.arrow(pos_truncated[0,i,n], pos_truncated[1,i,n], p_truncated[0,i,n]/2,\
                   p_truncated[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
        axes.arrow(pos_truncated[0,i,n], pos_truncated[1,i,n], -p_truncated[0,i,n]/2,\
                   -p_truncated[1,i,n]/2, head_width=0.1, head_length=0.1,\
                       fc='k', ec='k', zorder = 11)
        
        s = 5 #spacer

        if plot == 'Q':
            CS = axes.contourf(x,y,umag, levels = levels, cmap = 'twilight_r', zorder  = 1)
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
    plt.pause(0.001)
    # plt.savefig(str(n) + '.png', bbox_inches = 'tight', dpi = 300)
    
    progress = (n)/T*100
    if progress in np.linspace(0,100,21):
        print('Progress is ' + str(progress) + '%')
        
        #%%
rmag_t = np.linspace(0,5,1000)
Us_t = Us*((np.exp(-beta*(rmag_t - D)))/(1+np.exp(-beta*(rmag_t - D))))

plt.plot(rmag_t, Us_t)
plt.show()