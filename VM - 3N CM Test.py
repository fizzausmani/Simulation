import numpy as np
import matplotlib.pyplot as plt
from scipy.special import yn, kn, struve
import pandas as pd
import random as rd
import time as time
import datetime
import math
import scipy.interpolate as interp

#%%
OS = 'M'

if OS == 'W':
    save_path = 'C:\\Users\\fusmani\\Box\\Research\\Python\\Full Results\\N3\\'
    
elif OS == 'M':
    save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/N3/'
    
#%% Defining parameters
T = 500
dt = 0.05
steps = int(T/dt)
t = np.linspace(0, T, int(steps)+1) # time array for plots etc
truncation_point = int(1/dt)
steps_truncated = int(steps/truncation_point)

# system parameters
F = 50 # magnitude of force for dipoles (+1 or -1 if nondimensionalized)
L = 0.02 # half dipole distance 
sd = 0.01 # Saffman-Delbruck length: large for 2D, small for 3D
H = 0.1 # for confined membranes
sigma_t = 1000 # brownian motion parameter

beta = 20
Us = 5
D = 1

#specify number of source particles, define if system is free or confined: 
    # 'F' for free and 'C' for confined
N = 3 #source particles (dipoles)
system = 'C'
test = 1
a = 10 # box dimension

z = 200
x = np.linspace(-a/2, a/2, z)
y = np.linspace(-a/2, a/2, z)
X,Y = np.meshgrid(x, y)
dx = x[1] - x[0]

saving_string = system + 'M MP test' + str(test) + ' T' + str(T) + ' N' + str(N) + ' sd' + str(sd)

if sd == 1000:
    di = np.logspace(-7, 0, 100000)
elif sd == 10:
    di = np.logspace(-5, 3, 100000)
elif sd == 1:
    di = np.logspace(-3, 5, 100000)
elif sd == 0.1:
    di = np.logspace(-2, 7, 100000)
elif sd == 0.01:
    di = np.logspace(-1, 9, 100000)

if system == 'F':
    Ar_f = np.pi*(struve(0,di)-struve(1,di)/di+2/(np.pi*di**2)-(yn(0,di)-yn(2,di))/2)
    Br_f = np.pi*(-struve(0,di)+(2*struve(1,di))/di-4/(np.pi*di**2)-yn(2,di))
    cs = interp.CubicSpline(di,Ar_f)
    cs2 = interp.CubicSpline(di,Br_f)
elif system == 'C':
    Ar_c = -2/(di**2)+2*kn(0,di)+2*kn(1,di)/di
    Br_c = 4/(di**2)-2*kn(0,di)-4*kn(1,di)/di
    cs = interp.CubicSpline(di,Ar_c)
    cs2 = interp.CubicSpline(di,Br_c)

pos = np.zeros([2,N,int(steps)+1])
p = np.zeros([2,N,int(steps)+1])

pos[:,0,0] = np.array([-2, 0])
pos[:,1,0] = np.array([0, 0])
pos[:,2,0] = np.array([-1, -1.5])

if test == 1:
    th = np.array([np.pi/4, 3*np.pi/4, np.pi/2])
elif test == 2:
    th = np.array([np.pi/4, 3*np.pi/4, 0])

for i in range(N):
    p[:,i,0] = np.array([np.cos(th[i]), np.sin(th[i])])
    pmag = np.sqrt(np.dot(p[:,i,0],p[:,i,0]))
    p[:,i,0] = p[:,i,0]/pmag
    # ori_sp0[:,i] = ori_i[:,i,-1]
    # pos_sp[:,i,0] = pos_sources_i[:,i,-1]

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
    mi = np.concatenate((mi_E, mi_W, mi_N, mi_S, mi_NE, mi_NW, mi_SE, mi_SW),\
                        axis = 1)
    return mi

#%% Time-marching loop
grid_pos = np.array([np.ravel(X),np.ravel(Y)])

x_g = grid_pos[0,:]
x_g = x_g[:,np.newaxis]*np.ones([N*2])
y_g = grid_pos[1,:]
y_g = y_g[:,np.newaxis]*np.ones([N*2])

ux_f = np.zeros([z, z, steps_truncated+1])
uy_f = np.zeros([z, z, steps_truncated+1])

def u_mesh(pos, p):
    px = p[0,:]
    px = np.repeat(px,2)
    # px = np.tile(px,9)
    px[1::2] = -1*px[1::2]
    
    py = p[1,:]
    py = np.repeat(py,2)
    # py = np.tile(py,9)
    py[1::2] = -1*py[1::2]
    
    # images = mi(pos[:,:])
    # x_pos = np.concatenate((pos[0,:],images[0,:]), axis = 0)
    x_pos = pos[0,:]
    x_center = np.repeat(x_pos,2)
    x_pos = np.repeat(x_pos,2)
    x_pos[0:-1:2] += L*px[0:-1:2]
    x_pos[1::2] += L*px[1::2]
    x_pos = x_pos[:,np.newaxis]*np.ones([z*z])
    
    # y_pos = np.concatenate((pos[1,:],images[1,:]), axis = 0)
    y_pos = pos[1,:]
    y_center = np.repeat(y_pos,2)
    y_pos = np.repeat(y_pos,2)
    y_pos[0:-1:2] += L*py[0:-1:2]
    y_pos[1::2] += L*py[1::2]
    y_pos = y_pos[:,np.newaxis]*np.ones([z*z])
    
    rmag_c = ((x_g-np.transpose(x_center))**2+(y_g-np.transpose(y_center))**2)**0.5
    selfcheck = (rmag_c>0.5)

    rx = (x_g-np.transpose(x_pos))
    ry = (y_g-np.transpose(y_pos))
    rxy = (x_g-np.transpose(x_pos))*(y_g-np.transpose(y_pos))
    rmag = (rx**2+ry**2)**0.5
    
    # d = rmag/sd
    d = rmag/(np.sqrt(sd*H))
    ar = cs(d)
    br = cs2(d)
    
    C11 = (ar + br/(rmag**2)*rx**2)*selfcheck
    C12 = ((br/(rmag**2))*rxy)*selfcheck
    C21 = C12
    C22 = (ar + br/(rmag**2)*ry**2)*selfcheck
    
    Ust_x = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))*\
             ((x_g-np.transpose(x_center))/rmag_c))
    Ust_y = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))*\
             ((y_g-np.transpose(y_center))/rmag_c))
    
    Ust_x = Ust_x*selfcheck
    Ust_y = Ust_y*selfcheck
    Ust_x = np.sum(Ust_x, axis = 1)
    Ust_y = np.sum(Ust_y, axis = 1)
    
    ux = (np.dot(C11,F*px) + np.dot(C12,F*py))/(4*np.pi)
    uy = (np.dot(C22,F*py) + np.dot(C21,F*px))/(4*np.pi)
    
    ux_st = ux + Ust_x
    uy_st = uy + Ust_y
    
    ux_st = np.reshape(ux_st, (z,z))
    uy_st = np.reshape(uy_st, (z,z))
    
    ux = np.reshape(ux,(z,z))
    uy = np.reshape(uy,(z,z))

    return ux, uy, ux_st, uy_st
    
def du_particle(pos, ux_st, uy_st):
    u_particle = np.zeros([2,N])
    ux_interp = interp.RectBivariateSpline(x, y, ux_st, kx = 3, ky = 3)
    uy_interp = interp.RectBivariateSpline(x, y, uy_st, kx = 3, ky = 3)

    for i in range(0,N):
        u_particle[0,i] = ux_interp(pos[1,i], pos[0,i])[0]
        u_particle[1,i] = uy_interp(pos[1,i], pos[0,i])[0]

    return u_particle

def dp_particle(ux, uy, pos, p):
    ux_interp = interp.RectBivariateSpline(x, y, ux, kx = 3, ky = 3)
    uy_interp = interp.RectBivariateSpline(x, y, uy, kx = 3, ky = 3)
    pdot = np.zeros([2,N])

    for i in range(0,N):
        u_r = uy_interp(pos[1,i], pos[0,i]+0.005)[0]
        u_l = uy_interp(pos[1,i], pos[0,i]-0.005)[0]
        u_t = ux_interp(pos[1,i]+0.005, pos[0,i])[0]
        u_b = ux_interp(pos[1,i]-0.005, pos[0,i])[0]
        duydx = (1/(2*(0.005)))*(u_r[0] - u_l[0])
        duxdy = (1/(2*(0.005)))*(u_t[0] - u_b[0])
        omega = duydx - duxdy
        pdot[:,i] = np.array([-omega*p[1,i], omega*p[0,i]])/2.

    return pdot

start_time = time.perf_counter()

for n in range(0, steps):
    temp_pos = np.zeros([2,N])
    temp_p = np.zeros([2,N]) 
    
    ux, uy, ux_st, uy_st = u_mesh(pos[:,:,n], p[:,:,n])
    u_particle = du_particle(pos[:,:,n], ux_st, uy_st)
    p_particle = dp_particle(ux_st, uy_st, pos[:,:,n], p[:,:,n])
    
    for i in range(0,N):
        dpbr = (1/(2*np.pi*dt*sigma_t))*np.random.normal(0,1,2) # brownian translational velocity
        dubr = (1/(2*np.pi*dt*sigma_t))*np.random.normal(0,1,2) # brownian rotational velocity

        temp_pos[:,i] = pos[:,i,n] + 0.5*dt*(u_particle[:,i] + dubr)
    
        temp_p[:,i] = p[:,i,n] + 0.5*dt*(p_particle[:,i] + dpbr)
        pmag = np.sqrt(np.dot(temp_p[:,i], temp_p[:,i]))
        temp_p[:,i] = temp_p[:,i]/pmag

        if temp_pos[0,i] > (a/2):
            temp_pos[0,i] += -a
        elif temp_pos[1,i] > (a/2):
            temp_pos[1,i] += -a
        elif temp_pos[0,i] < -(a/2):
            temp_pos[0,i] += a
        elif temp_pos[1,i] < -(a/2):
            temp_pos[1,i] += a
        
    ux_temp, uy_temp, ux_st_temp, uy_st_temp = u_mesh(temp_pos, temp_p)
    u_particle_temp = du_particle(temp_pos, ux_st_temp, uy_st_temp)
    p_particle_temp = dp_particle(ux_st_temp, uy_st_temp, temp_pos, temp_p)
    
    for i in range(0,N):
        dpbr = (1/(2*np.pi*dt*sigma_t))*np.random.normal(0,1,2)*0 # brownian translational velocity
        dubr = (1/(2*np.pi*dt*sigma_t))*np.random.normal(0,1,2)*0 # brownian rotational velocity
        
        pos[:,i,n+1] = pos[:,i,n] + dt*((u_particle[:,i] + u_particle_temp[:,i]) + dubr)

        if pos[0,i,n+1] > (a/2):
            pos[0,i,n+1] += -a
        elif pos[1,i,n+1] > (a/2):
            pos[1,i,n+1] += -a
        elif pos[0,i,n+1] < -(a/2):
            pos[0,i,n+1] += a
        elif pos[1,i,n+1] < -(a/2):
            pos[1,i,n+1] += a

        p[:,i,n+1] = p[:,i,n] + dt*((p_particle[:,i] + p_particle_temp[:,i]) + dpbr)
        pmag = np.sqrt(np.dot(p[:,i,n+1], p[:,i,n+1]))
        p[:,i,n+1] = p[:,i,n+1]/pmag

    t_save = np.arange(0, steps, int(1/dt))
    
    if n in t_save:
        ux = (ux + ux_temp)/2
        uy = (uy + uy_temp)/2
        ux_f[:,:,math.floor(n*dt)] = ux[:,:]
        uy_f[:,:,math.floor(n*dt)] = uy[:,:]

    progress = (n*dt)/T*100
    if progress in np.linspace(0,100,11):
        print('Progress ' + str(round(progress,2)) + '% ' + str(datetime.datetime.now()))
        np.save(save_path + saving_string + ' ux', ux_f)
        np.save(save_path + saving_string + ' uy', uy_f)
        np.save(save_path + saving_string + ' pos', pos)
        np.save(save_path + saving_string + ' p', p)

end_time = time.perf_counter()
elapsed_time = end_time-start_time

 #%% Saving Arrays
df = pd.DataFrame([['T', T], ['dt', dt], ['Steps', steps], ['Box dimension, a', a], 
                   ['Grid spacing', dx], ['Meshgrid size, z', z],
                   ['Number of particles, SP', N], ['System, free or confined', system], 
                   ['sd', sd], ['Confinement height, H', H],
                   ['Force magnitude', F], ['Dipole length', L], ['Sigma_t', sigma_t],
                   ['Beta', beta], ['Us', Us],
                   ['Truncated steps', steps_truncated], 
                   ['Truncation point', truncation_point],
                   ['Run time', elapsed_time]])

np.save(save_path + saving_string + ' ux', ux_f)
np.save(save_path + saving_string + ' uy', uy_f)
np.save(save_path + saving_string + ' pos', pos)
np.save(save_path + saving_string + ' p', p)
df.to_csv(save_path + saving_string + ' ic.csv', index = False)

#%%
pos_t = pos[:,:,::truncation_point]
p_t = p[:,:,::truncation_point]

# plot = 'Q'
plot = 'S'

figure, axes = plt.subplots(figsize=(10,8))

# n = 0
# if n == 0:
for n in range(0, T):
    plt.cla()
    for i in range(N):
        umag = np.sqrt(ux_f[:,:,n]**2+uy_f[:,:,n]**2)
        
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
            plt.streamplot(X[::s,::s], Y[::s,::s], ux_f[::s,::s,n], uy_f[::s,::s,n],\
                           color = 'k')
            
        axes.add_artist(circle)
        
    if n == 0:
        cbar = figure.colorbar(CS, pad = 0.08)
    else:
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
        
# %%
figure, axes = plt.subplots(figsize=(10,8))

# n = 0
# if n == 0: 
for n in range(0, steps+1, 20):
    plt.cla()
    for i in range(N):
        
        axes.set_aspect(1)
        
        circle = plt.Circle((pos[0,i,n], pos[1,i,n]), 0.5, edgecolor = 'k',\
                            facecolor = 'Gainsboro', zorder = 10)
        axes.arrow(pos[0,i,n], pos[1,i,n], p[0,i,n]/2, p[1,i,n]/2, head_width=0.1,\
                   head_length=0.1, fc='k', ec='k', zorder = 11)
        axes.arrow(pos[0,i,n], pos[1,i,n], -p[0,i,n]/2, -p[1,i,n]/2, head_width=0.1,\
                   head_length=0.1, fc='k', ec='k', zorder = 11)
        
        axes.add_artist(circle)
    
    axes.axis([-a/2, a/2, -a/2, a/2])
    axes.tick_params(top = True, labeltop = True, right = True, labelright = True)
    
    plt.title('%3d' %(n*dt))
    # plt.pause(0.5)
    plt.savefig(str(n) + ' SP'  +str(N) + '.png', bbox_inches = 'tight')
