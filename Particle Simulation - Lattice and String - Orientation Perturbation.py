"""
This is a particle simulation code - meshgrid velocities are not
calculated. This code does not yet contain Brownian motion. 
"""
#%%
import os #Import os before importing numpy
#These next lines of code will tell numpy to explicitly use the specified number of cores
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import yn, kn, struve
import pandas as pd
import random as rd
from time import time
import math
import scipy.interpolate as interp
import datetime

#%% Defining parameters
T = 5000
dt = 0.05
steps = int(T/dt)
t = np.linspace(0, T, int(steps)+1) # time array for plots etc
truncation_point = int(1/dt)
steps_truncated = int(steps/truncation_point)

""" Fixed system and steric parameters (force magnitude, dipole distance, 
confinement height)"""
# system parameters
F = 50 # magnitude of force for dipoles (+1 or -1 if nondimensionalized)
L = 0.02 # half dipole distance 
H = 0.1 # for confined membranes

# steric parameters
beta = 20
Us = 5
D = 1

""" System parameters to vary below (number of particles, saffman-delbruck 
length, geometry, and box dimension)"""
N = 625 #source particles (dipoles)
sd = 0.1
system = 'C' # geometry - free or confined
a = 50 # box dimension

""" Based on the saffman-delbruck length, the interpolation range is set up. """
if sd == 1000 or sd == 100:
    di = np.logspace(-7, 0, 100000)
elif sd == 10:
    di = np.logspace(-5, 3, 100000)

elif sd == 1:
    di = np.logspace(-3, 5, 100000)
elif sd == 0.1:
    di = np.logspace(-2, 7, 100000)

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

""" The lines below setup orientations for nearly aligned simulations. """
# th = np.pi/2 # for nearly vertically aligned
# th = 0 # for nearly horizontally aligned
eps = 0.0 # perturbation 
for i in range(N):
    th = rd.uniform(0,2*np.pi)
    p[:,i,0] = np.array([np.cos(th)+rd.uniform(-eps, eps),\
                         np.sin(th)+rd.uniform(-eps, eps)]) 
    pmag = np.sqrt(np.dot(p[:,i,0],p[:,i,0]))
    p[:,i,0] = p[:,i,0]/pmag

""" The lines below are for the test system with 2 particles.
N must be changed earlier in the code, to ensure the position
and orientation arrays are of the correct size. """
# p[:,0,0] = np.array([0.707,0.707])
# p[:,1,0] = np.array([1,0])
# pos[:,:,0] = np.array([[-2,0],
#                         [0,0]])

""" Function for setting up initial lattice-like positions.
Takes as input number of particles, returns their positions as a
square lattice. """
def lattice(N):
    bands = int(np.sqrt(N))
    coords = np.linspace(-a/2, a/2, 2*bands)
    fracs = np.arange(0,1+1/(2*bands), round(1/bands,3))

    pos0 = np.zeros([2,N])

    for i in range(0,bands):
        pos0[1, int(N*fracs[i]):int(N*fracs[i+1])] = (i==0)*coords[i] \
            + (i > 0)*coords[i*2]
        pos0[0, int(N*fracs[i]):int(N*fracs[i+1])] = coords[::2]
    
    return pos0

""" Function for setting up initial string positions. 
Takes as input number of particles, returns their positions as a
# string. """
def string(N):
    pos0 = np.zeros([2,N])
    pos0[0,:] = -a
    dx = 2
    for i in range(1,N):
        pos0[0,i] = pos0[0,i-1] + dx
    return pos0

#%% String or lattice selection
""" Specify test system type, string or lattice"""
test_system = 'L' # 'S' for string, 'L' for lattice

if test_system == 'S':
    pos[:,:,0] = string(N)
    saving_string = system + 'M T' + str(T) + ' N' + str(N) + ' string sd' +\
            str(sd) + ' th' + str(round(th,3))
elif test_system == 'L':
    pos[:,:,0] = lattice(N)
    saving_string = system + 'M T' + str(T) + ' N' + str(N) + ' lattice sd' +\
            str(sd) + ' isotropic'

""" Specify save path below. """
# save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/'
save_path = 'C:\\Users\\fusmani\\Box\\Research\\Python\\Stability Analysis\\'

# %% Functions for translational and rotational velocities

def du_particle(pos, p):
    """ The lines below create the array with the x and y coordinates of the 
    target points:
    the particle centers. x_targets and y_targets should be T x S shaped, 
    where T = number of targets (T = N), and S = number of sources (2N). """
    x_targets = pos[0,:] 
    x_targets = x_targets[:,np.newaxis]*np.ones([N*2]) 
    y_targets = pos[1,:]
    y_targets = y_targets[:,np.newaxis]*np.ones([N*2])

    """ px and py are the orientations for the source points.
    Add in the third line for both if running a periodic simulation. """
    px = p[0,:]
    px = np.repeat(px,2)
    # px = np.tile(px,9) # uncomment if setting up a periodic simulation
    px[1::2] = -1*px[1::2]
    
    py = p[1,:]
    py = np.repeat(py,2)
    # py = np.tile(py,9) # uncomment if setting up a periodic simulation
    py[1::2] = -1*py[1::2]
    
    """ The lines below create the array with the x and y coordinates of the 
    source points. 
    x_sources and y_sources should be S x T shaped.
    The commented lines should be added in if running a periodic simulation. """
    x_sources = pos[0,:]
    # images = mi(pos[:,:]) # uncomment if setting up a periodic simulation
    # x_sources = np.concatenate((pos[0,:],images[0,:]), axis = 0) # uncomment if setting up a periodic simulation
    x_center = np.repeat(x_sources,2) # needed for self-check 
    x_sources = np.repeat(x_sources,2)
    x_sources[0:-1:2] += L*px[0:-1:2]
    x_sources[1::2] += L*px[1::2]
    x_sources = x_sources[:,np.newaxis]*np.ones(N)
    
    y_sources = pos[1,:]
    # y_sources = np.concatenate((pos[1,:],images[1,:]), axis = 0) # uncomment if setting up a periodic simulation
    y_center = np.repeat(y_sources,2) # needed for self-check 
    y_sources = np.repeat(y_sources,2)
    y_sources[0:-1:2] += L*py[0:-1:2]
    y_sources[1::2] += L*py[1::2]
    y_sources = y_sources[:,np.newaxis]*np.ones(N)
    
    # Sets up self-check
    rmag_c = ((x_targets-np.transpose(x_center))**2\
              +(y_targets-np.transpose(y_center))**2)**0.5
    selfcheck = (rmag_c>0.5)

    rx = (x_targets-np.transpose(x_sources))
    ry = (y_targets-np.transpose(y_sources))
    rxy = (x_targets-np.transpose(x_sources))*(y_targets-np.transpose(y_sources))
    rmag = (rx**2+ry**2)**0.5
    
    # d = rmag/sd
    d = rmag/np.sqrt(sd*H)
    ar = cs(d)
    br = cs2(d)
    
    C11 = (ar + br/(rmag**2)*rx**2)*selfcheck
    C12 = ((br/(rmag**2))*rxy)*selfcheck
    C21 = C12
    C22 = (ar + br/(rmag**2)*ry**2)*selfcheck
    
    """ Steric velocity calculation. the np.isnan function is used as
    'False' does not yield a 0 when working with nan. Steric velocity
    should be commented out during 2 particle test simulation, to 
    correctly test system behavior. """
    Ust_x = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))\
             *((x_targets-np.transpose(x_center))/rmag_c))
    Ust_x[np.isnan(Ust_x)] = 0
    Ust_y = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))\
             *((y_targets-np.transpose(y_center))/rmag_c))
    Ust_y[np.isnan(Ust_y)] = 0

    Ust_x = np.sum(Ust_x, axis = 1)
    Ust_y = np.sum(Ust_y, axis = 1)
    
    ux = (np.dot(C11,F*px) + np.dot(C12,F*py))/(4*np.pi)
    uy = (np.dot(C22,F*py) + np.dot(C21,F*px))/(4*np.pi)
    
    ux_st = ux + Ust_x
    uy_st = uy + Ust_y

    return ux, uy, ux_st, uy_st

def dp_particle(pos,p):
    """ The lines below create the array with the x and y coordinates of the 
    target points:
    4 points to the right, left, top, and bottom of the particle centers.
    The first two points are changed in the x-direction, and the next two in 
    the y-direction.
    The order for the target points then is: right, left, top, bottom. """
    x_targets = pos[0,:]
    x_targets = np.repeat(x_targets,4)
    x_targets[0::4] = x_targets[0::4] + 0.005
    x_targets[1::4] = x_targets[1::4] - 0.005
    x_targets = x_targets[:,np.newaxis]*np.ones([N*2])

    y_targets = pos[1,:]
    y_targets = np.repeat(y_targets,4)
    y_targets[2::4] = y_targets[2::4] + 0.005
    y_targets[3::4] = y_targets[3::4] - 0.005
    y_targets = y_targets[:,np.newaxis]*np.ones([N*2])
    
    """ px and py are the orientations for the source points. They remain the 
    same as before. """
    px = p[0,:]
    px = np.repeat(px,2)
    # px = np.tile(px,9)
    px[1::2] = -1*px[1::2]

    py = p[1,:]
    py = np.repeat(py,2)
    # py = np.tile(py,9)
    py[1::2] = -1*py[1::2]

    """ Below are the source arrays. They are formulated the same way as in 
    u_particle;
    only the size changes (as there are 4N target points now instead of N). """
    # images = mi(pos[:,:])
    # x_sources = np.concatenate((pos[0,:],images[0,:]), axis = 0)
    x_sources = pos[0,:]
    x_center = np.repeat(x_sources,2)
    x_sources = np.repeat(x_sources,2)
    x_sources[0:-1:2] += L*px[0:-1:2]
    x_sources[1::2] += L*px[1::2]
    x_sources = x_sources[:,np.newaxis]*np.ones([N*4])
    
    # y_sources = np.concatenate((pos[1,:],images[1,:]), axis = 0)
    y_sources = pos[1,:]
    y_center = np.repeat(y_sources,2)
    y_sources = np.repeat(y_sources,2)
    y_sources[0:-1:2] += L*py[0:-1:2]
    y_sources[1::2] += L*py[1::2]
    y_sources = y_sources[:,np.newaxis]*np.ones([N*4])
    
    rmag_c = ((x_targets-np.transpose(x_center))**2+\
              (y_targets-np.transpose(y_center))**2)**0.5
    selfcheck = (rmag_c>0.5)

    rx = (x_targets-np.transpose(x_sources))
    ry = (y_targets-np.transpose(y_sources))
    rxy = (x_targets-np.transpose(x_sources))*(y_targets-np.transpose(y_sources))
    rmag = (rx**2+ry**2)**0.5
    
    # d = rmag/sd
    d = rmag/np.sqrt(sd*H)
    ar = cs(d)
    br = cs2(d)
    
    C11 = (ar + br/(rmag**2)*rx**2)*selfcheck
    C12 = ((br/(rmag**2))*rxy)*selfcheck
    C21 = C12
    C22 = (ar + br/(rmag**2)*ry**2)*selfcheck
    
    Ust_x = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))\
             *((x_targets-np.transpose(x_center))/rmag_c))
    Ust_x[np.isnan(Ust_x)] = 0
    Ust_y = (Us*((np.exp(-beta*(rmag_c - D)))/(1+np.exp(-beta*(rmag_c - D))))\
             *((y_targets-np.transpose(y_center))/rmag_c))
    Ust_y[np.isnan(Ust_y)] = 0

    Ust_x = np.sum(Ust_x, axis = 1)
    Ust_y = np.sum(Ust_y, axis = 1)
    
    ux = (np.dot(C11,F*px) + np.dot(C12,F*py))/(4*np.pi)
    uy = (np.dot(C22,F*py) + np.dot(C21,F*px))/(4*np.pi)
    
    ux_st = ux + Ust_x
    uy_st = uy + Ust_y
    
    # ux = (np.dot(C11,F*px) + np.dot(C12,F*py))/(4*np.pi)
    # uy = (np.dot(C22,F*py) + np.dot(C21,F*px))/(4*np.pi)

    pdot = np.zeros([2,N])

    """ The velocities are extracted such that u_r and u_l are the y-components 
    of the velocities right and left of the target point, and u_t and u_b are the 
    x-components of the velocities above and below the target point. 
    They are extracted in the roder they were set up in, at the beginning of 
    the function. """
    u_r = uy_st[0::4]
    u_l = uy_st[1::4]
    u_t = ux_st[2::4]
    u_b = ux_st[3::4]

    for i in range(0,N):
        duydx = (1/(2*(0.005)))*(u_r[i] - u_l[i])
        duxdy = (1/(2*(0.005)))*(u_t[i] - u_b[i])
        omega = duydx - duxdy
        pdot[:,i] = np.array([-omega*p[1,i], omega*p[0,i]])/2.

    return pdot

#%% Time-advancing position and orientation

u = np.zeros([2,N,steps_truncated+1])
dp = np.zeros([2,N,steps_truncated+1])

start_time=time()

""" The time-advancing is done via Heun's method currently,
verified to work correctly for the test system. """

for n in range(0, steps):
    ux, uy, ux_st, uy_st = du_particle(pos[:,:,n], p[:,:,n])
    p_particle = dp_particle(pos[:,:,n], p[:,:,n])

    temp_pos = np.zeros([2,N])
    temp_p = np.zeros([2,N])
    
    temp_pos = pos[:,:,n] + dt*np.array([ux_st,uy_st])

    for i in range(0,N):
        temp_p[:,i] = p[:,i,n] + dt*p_particle[:,i]
        pmag = np.sqrt(np.dot(temp_p[:,i], temp_p[:,i]))
        temp_p[:,i] = temp_p[:,i]/pmag

    ux_temp, uy_temp, ux_st_temp, uy_st_temp = du_particle(temp_pos, temp_p)
    p_particle_temp = dp_particle(temp_pos, temp_p)

    pos[:,:,n+1] = pos[:,:,n] + dt*np.array([(ux_st + ux_st_temp)/2,\
                                             (uy_st + uy_st_temp)/2])

    for i in range(0,N):
        p[:,i,n+1] = p[:,i,n] + dt*(p_particle[:,i] + p_particle_temp[:,i])/2
        pmag = np.sqrt(np.dot(p[:,i,n+1], p[:,i,n+1]))
        p[:,i,n+1] = p[:,i,n+1]/pmag
    
    t_save = np.arange(0, steps, int(1/dt))
    
    """ The velocities are saved at every time T instead of every timestep. """
    if n in t_save:
        u[:,:,math.floor(n*dt)] = np.array([(ux_st + ux_st_temp)/2,\
                                            (uy_st + uy_st_temp)/2])
        dp[:,:,math.floor(n*dt)] = np.array([p_particle + p_particle_temp])/2
       
    progress = (n*dt)/T*100
    if progress in np.linspace(0,100,11):
        print('Progress ' + str(round(progress,2)) + '% ' + str(datetime.datetime.now()))
        np.save(save_path + saving_string + ' u', u)
        np.save(save_path + saving_string + ' dp', dp)
        np.save(save_path + saving_string + ' pos', pos)
        np.save(save_path + saving_string + ' p', p)

end_time=time()
elapsed_time = end_time-start_time

#%% Saving routin

df = pd.DataFrame([['T', T], ['dt', dt], ['Steps', steps], ['Initial box dimension, a', a],
                   ['Number of particles, N', N], ['System, free or confined', system], 
                   ['sd', sd], ['Confinement height, H', H],
                   ['Force magnitude', F], ['Dipole length', L],
                   ['Beta', beta], ['Us', Us],
                   ['Truncated steps', steps_truncated], ['Truncation point', truncation_point],
                   ['Run time', elapsed_time]])

""" Saving string contains information about the geometry, the time the
simulation is run for, number of particles, the SD length, and type of test run
 (lattice or string, and initial orientation). """

np.save(save_path + saving_string + ' u', u)
np.save(save_path + saving_string + ' dp', dp)
np.save(save_path + saving_string + ' pos', pos)
np.save(save_path + saving_string + ' p', p)
df.to_csv(save_path + saving_string + ' ic.csv', index = False) 

# %% Plotting
""" Specify save path below. """

save_path_plots = '/Users/fizzausmani/Downloads/Lattice/N625/'

if test_system == 'S':
    fig, ax1 = plt.subplots(figsize=(7, 4))
elif test_system == 'L':
    fig, ax1 = plt.subplots(figsize=(5, 5))

bands = int(np.sqrt(N))

def plotlines(n,N):
    bands = int(np.sqrt(N))
    
    for i in range(0,bands+1):
        plt.plot(pos[0, i*bands:(i+1)*bands, n], pos[1, i*bands:(i+1)*bands, n])

# n = 0
# if n == 0:
for n in range(0, steps, 1000):
    plt.cla()
    for i in range(N):
        ax1.set_aspect(1)
        circle = plt.Circle((pos[0,i,n], pos[1,i,n]), 0.5, edgecolor = 'r', \
                            facecolor = 'w')
        ax1.add_artist(circle)
        
        ax1.arrow(pos[0,i,n], pos[1,i,n], p[0,i,n]/2, p[1,i,n]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')
        ax1.arrow(pos[0,i,n], pos[1,i,n], -p[0,i,n]/2, -p[1,i,n]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')
    
    # plotlines(n,N)

    ax1.axis([-30, 30, -30, 30]) # use for lattice simulations
    ax1.tick_params(top = True, labeltop = True, right = True, labelright = True)
    
    plt.title('%3d' %(n*dt))

    # plt.pause(0.001)
    plt.savefig(save_path_plots + str(n) + ' N'  + str(N) + '.png')

#%% Code for adding kinematic periodicity
""" The function below creates mirror images.
It takes as input the positions of the main particles,
and returns the positions of the image particles only. """
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
    mi = np.concatenate((mi_E, mi_W, mi_N, mi_S, mi_NE, mi_NW, mi_SE, mi_SW), 
                        axis = 1)
    return mi

""" The code below is for adding kinematic periodicity in the system.
temp_pos should be added in the first particle loop, and 
pos should be added in the second particle loop. """ 

        if temp_pos[0,i] > (a/2):
            temp_pos[0,i] += -a
        elif temp_pos[1,i] > (a/2):
            temp_pos[1,i] += -a
        elif temp_pos[0,i] < -(a/2):
            temp_pos[0,i] += a
        elif temp_pos[1,i] < -(a/2):
            temp_pos[1,i] += a

        if pos[0,i,n+1] > (a/2):
            pos[0,i,n+1] += -a
        elif pos[1,i,n+1] > (a/2):
            pos[1,i,n+1] += -a
        elif pos[0,i,n+1] < -(a/2):
            pos[0,i,n+1] += a
        elif pos[1,i,n+1] < -(a/2):
            pos[1,i,n+1] += a