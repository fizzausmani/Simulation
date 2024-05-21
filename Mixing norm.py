#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import glob as glob
import os

#%% Importing concentration data
path_f1000 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/sd1000'
path_f10 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/sd10'
path_f1 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/sd1'
path_f01 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/fsd01'
# path_c01 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/csd01'
# path_c1 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/csd1'

preamble = 'AD T1000 dt0.05 SP4 FM conc seed'
pattern = preamble + '*.npy'
c_f1000 = glob.glob(f'{path_f1000}/{pattern}')
c_f10 = glob.glob(f'{path_f10}/{pattern}')
c_f1 = glob.glob(f'{path_f1}/{pattern}')
c_f01 = glob.glob(f'{path_f01}/{pattern}')
# c_c01 = glob.glob(f'{path_c01}/{pattern}')

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

# def multinorm(conc_path, steps, a, z):
#     conc_dic = {os.path.basename(file): np.load(file) for file in conc_path}
#     c1 = conc_dic[preamble + '0.npy']
#     c2 = conc_dic[preamble + '1.npy']
#     c3 = conc_dic[preamble + '2.npy']
#     c4 = conc_dic[preamble + '3.npy']
#     c5 = conc_dic[preamble + '4.npy']
#     c6 = conc_dic[preamble + '5.npy']

#     s1 = np.zeros(steps)#, dtype = complex)
#     s2 = np.zeros(steps)#, dtype = complex)
#     s3 = np.zeros(steps)#, dtype = complex)
#     s4 = np.zeros(steps)#, dtype = complex)
#     s5 = np.zeros(steps)#, dtype = complex)
#     s6 = np.zeros(steps)#, dtype = complex)

#     kx = np.linspace(0,z,z)/a
#     ky = np.linspace(0,z,z)/a
#     KX, KY = np.meshgrid(kx,ky)

#     times = np.arange(0,steps,1)
#     for n in times:
#         fft_c1 = fft2(c1[:,:,n])
#         fft_c2 = fft2(c2[:,:,n])
#         fft_c3 = fft2(c3[:,:,n])
#         fft_c4 = fft2(c4[:,:,n])
#         fft_c5 = fft2(c5[:,:,n])
#         fft_c6 = fft2(c6[:,:,n])

#         s1[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c1))**2))
#         s2[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c2))**2))
#         s3[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c3))**2))
#         s4[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c4))**2))
#         s5[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c5))**2))
#         s6[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c6))**2))

#     s1 = s1/s1[0]
#     s2 = s2/s2[0]
#     s3 = s3/s3[0]
#     s4 = s4/s4[0]
#     s5 = s5/s5[0]
#     s6 = s6/s6[0]
#     s = np.average((s1, s2, s3, s4, s5, s6), axis = 0)

#     return s

def multinorm(conc_path, steps, a, z):
    conc_dic = {os.path.basename(file): np.load(file) for file in conc_path}
    
    s_values = []
    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)
    times = np.arange(0,steps,1)

    for i in range(len(conc_dic)):
        c = conc_dic[preamble + str(i) + '.npy']
        # if not c.all():
        #     i + 1
        s = np.zeros(steps)
        for n in times:
            fft_c = fft2(c[:,:,n])
            s[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_c))**2))
        s = s/s[0]
        s_values.append(s)

    s = np.average(s_values, axis = 0)

    return s

#%%
s_F4_l1000 = multinorm(c_f1000, steps1, a2, z2)
s_F4_l10 = multinorm(c_f10, steps1, a2, z2)
s_F4_l1 = multinorm(c_f1, steps1, a2, z2)
s_F4_l01 = multinorm(c_f01, steps1, a2, z2)
# s_C4_l01 = multinorm(c_c01, steps1, a1, z2)

#%%
plt.cla()
plt.plot(times, s_F4_l1000, 'k')
plt.plot(times, s_F4_l10, 'r')
plt.plot(times, s_F4_l1, 'salmon')
plt.plot(times, s_F4_l01, 'coral')
# plt.plot(times, s_C4_l01, 'b_')

plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)

# dash_patch = mlines.Line2D([], [], color = 'k', marker='None', linestyle='None',
#                           markersize=10, label='10x10 box')
# o_patch = mlines.Line2D([], [], color = 'k', marker='.', linestyle='None',
#                           markersize=10, markerfacecolor = 'white', label='20x20 box')

patch1000 = mlines.Line2D([], [], color = 'k', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 1000$')
patch10 = mlines.Line2D([], [], color = 'r', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 10$')
patch1 = mlines.Line2D([], [], color = 'salmon', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 1$')
patch01 = mlines.Line2D([], [], color = 'coral', marker='_', linestyle='None',
                          markersize=10, label = '$\ell = 0.1$')

plt.legend(handles=[patch01, patch1, patch10, patch1000], fontsize = 14)
plt.savefig('N4.png', bbox_inches = 'tight', dpi = 300)
plt.show()


# %%
