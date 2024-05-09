#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import glob as glob
import os

#%% Importing concentration data
path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/sd1000'
pattern = 'AD RK4 T1000 dt0.05 SP4 FM conc seed*.npy'
preamble = 'AD RK4 T1000 dt0.05 SP4 FM conc seed'
# pattern = preamble + '*.npy'
c = glob.glob(f'{path}/{pattern}')
arrays =  {os.path.basename(file): np.load(file) for file in c}

#%%
c1 = arrays[preamble + '0.npy']
c2 = arrays[preamble + '1.npy']
c3 = arrays[preamble + '2.npy']
c4 = arrays[preamble + '3.npy']
c5 = arrays[preamble + '4.npy']
c6 = arrays[preamble + '5.npy']

#%% Initiation
steps1 = 1000
T1 = 1000
a1 = 10
a2 = 20
z1 = 100
z2 = 200
times = np.arange(0,steps1,1)

#%% Norm functions
def norm(conc, steps, a, z):
    s = np.zeros(steps)#, dtype = complex)

    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)

    times = np.arange(0,steps,1)
    for n in times:
        fft_conc = fft2(conc[:,:,n])
        s[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc))**2))
    s = s/s[0]

    return s

def binorm(conc1, conc2, steps, a, z):
    s1 = np.zeros(steps)#, dtype = complex)
    s2 = np.zeros(steps)#, dtype = complex)

    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)

    times = np.arange(0,steps,1)
    for n in times:
        fft_conc1 = fft2(conc1[:,:,n])
        fft_conc2 = fft2(conc2[:,:,n])
        s1[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc1))**2))
        s2[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc2))**2))

    s1 = s1/s1[0]
    s2 = s2/s2[0]
    s = (s1 + s2)/2

    return s

def trinorm(conc1, conc2, conc3, steps, a, z):
    s1 = np.zeros(steps)#, dtype = complex)
    s2 = np.zeros(steps)#, dtype = complex)
    s3 = np.zeros(steps)#, dtype = complex)

    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)

    times = np.arange(0,steps,1)
    for n in times:
        fft_conc1 = fft2(conc1[:,:,n])
        fft_conc2 = fft2(conc2[:,:,n])
        fft_conc3 = fft2(conc3[:,:,n])
        s1[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc1))**2))
        s2[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc2))**2))
        s3[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc3))**2))

    s1 = s1/s1[0]
    s2 = s2/s2[0]
    s3 = s3/s3[0]
    s = (s1 + s2 + s3)/3

    return s

def multinorm(conc1, conc2, conc3, conc4, conc5, steps, a, z):
    s1 = np.zeros(steps)#, dtype = complex)
    s2 = np.zeros(steps)#, dtype = complex)
    s3 = np.zeros(steps)#, dtype = complex)
    s4 = np.zeros(steps)#, dtype = complex)
    s5 = np.zeros(steps)#, dtype = complex)

    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)

    times = np.arange(0,steps,1)
    for n in times:
        fft_conc1 = fft2(conc1[:,:,n])
        fft_conc2 = fft2(conc2[:,:,n])
        fft_conc3 = fft2(conc3[:,:,n])
        fft_conc4 = fft2(conc4[:,:,n])
        fft_conc5 = fft2(conc5[:,:,n])

        s1[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc1))**2))
        s2[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc2))**2))
        s3[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc3))**2))
        s4[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc4))**2))
        s5[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc5))**2))

    s1 = s1/s1[0]
    s2 = s2/s2[0]
    s3 = s3/s3[0]
    s = (s1 + s2 + s3)/3

    return s

#%%
s_F4_l1000 = norm(c1, steps1, a2, z2)
s_F4_l10 = norm(c2, steps1, a2, z2)
s_F4_l1 = norm(c3, steps1, a2, z2)
s_F4_l01 = norm(c4, steps1, a2, z2)
s_C4_l01 = norm(c5, steps1, a1, z2)

#%%
plt.cla()
plt.plot(times, s_F4_l1000, 'k.')
plt.plot(times, s_F4_l10, 'r.')
plt.plot(times, s_F4_l1, 'c.')
plt.plot(times, s_F4_l01, 'g.')
plt.plot(times, s_C4_l01, 'b_')

plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)

dash_patch = mlines.Line2D([], [], color = 'k', marker='None', linestyle='None',
                          markersize=10, label='10x10 box')
o_patch = mlines.Line2D([], [], color = 'k', marker='.', linestyle='None',
                          markersize=10, markerfacecolor = 'white', label='20x20 box')

black_patch = mlines.Line2D([], [], color = 'k', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 1000$')
red_patch = mlines.Line2D([], [], color = 'r', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 10$')
cyan_patch = mlines.Line2D([], [], color = 'c', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 1$')
green_patch = mlines.Line2D([], [], color = 'g', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 0.1$')

plt.legend(handles=[dash_patch, o_patch, black_patch, red_patch, cyan_patch, green_patch], fontsize = 14)

plt.show()

plt.savefig('4v16.png', bbox_inches = 'tight', dpi = 300)