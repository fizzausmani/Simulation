#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import glob as glob

#%% Importing concentration data
path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/0/Mixing'
pattern = '*.npy'
conc = glob.glob(f'{path}/{pattern}')

#%% Initiation
steps1 = 1000
T1 = 1000
a1 = 10
a2 = 20
times = np.arange(0,steps1,1)

#%% Norm functions
def norm(conc, steps, a):
    s = np.zeros(steps)#, dtype = complex)

    if a == 10:
        z = 100
    elif a == 20:
        z = 200

    kx = np.linspace(0,z,z)/a
    ky = np.linspace(0,z,z)/a
    KX, KY = np.meshgrid(kx,ky)

    times = np.arange(0,steps,1)
    for n in times:
        fft_conc = fft2(conc[:,:,n])
        s[n] = np.sqrt(np.sum((1/np.sqrt(1+(4*np.pi**2)*(KX**2+KY**2)))*np.abs((fft_conc))**2))
    s = s/s[0]

    return s

def binorm(conc1, conc2, steps, a):
    s1 = np.zeros(steps)#, dtype = complex)
    s2 = np.zeros(steps)#, dtype = complex)

    if a == 10:
        z = 100
    elif a == 20:
        z = 200

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

def trinorm(conc1, conc2, conc3, steps, a):
    s1 = np.zeros(steps)#, dtype = complex)
    s2 = np.zeros(steps)#, dtype = complex)
    s3 = np.zeros(steps)#, dtype = complex)

    if a == 10:
        z = 100
    elif a == 20:
        z = 200

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
s_F4_l1000_a10_1 = trinorm(F4_l1000_10x10_1, F4_l1000_10x10_2, F4_l1000_10x10_3, steps1, a2)
s_F4_l1000_a10_2 = binorm(F4_l1000_10x10_4, F4_l1000_10x10_5, steps1, a1)
s_F4_l1000_a10 = (s_F4_l1000_a10_1+s_F4_l1000_a10_2)/2

s_F16_l1000 = multinorm(F16_l1000_1, F16_l1000_2, F16_l1000_3, F16_l1000_4, F16_l1000_5, steps1, a2)

# %%
s_F4_l10_a10_1 = trinorm(F4_l10_10x10_1, F4_l10_10x10_2, F4_l10_10x10_3, steps1, a2)
s_F4_l10_a10_2 = binorm(F4_l10_10x10_4, F4_l10_10x10_5, steps1, a1)
s_F4_l10_a10 = (s_F4_l10_a10_1+s_F4_l10_a10_2)/2

s_F16_l10 = multinorm(F16_l10_1, F16_l10_2, F16_l10_3, F16_l10_4, F16_l10_5, steps1, a2)
# s_F16_l10 = norm(F16_l10_1,steps1, a2)

#%%
s_F4_l1_a10_1 = trinorm(F4_l1_10x10_2, F4_l1_10x10_4, F4_l1_10x10_5, steps1, a1)
s_F4_l1_a10_2 = norm(F4_l1_10x10_1, steps1, a2)
s_F4_l1_a10 = (s_F4_l1_a10_1+s_F4_l1_a10_2)/2

s_F16_l1 = multinorm(F16_l1_1, F16_l1_2, F16_l1_3, F16_l1_4, F16_l1_5, steps1, a2)

#%%
s_F4_l01_a10_1 = binorm(F4_l01_10x10_4, F4_l01_10x10_5, steps1, a1)
s_F4_l01_a10_2 = norm(F4_l01_10x10_1, steps1, a2)
s_F4_l01_a10 = (s_F4_l01_a10_1+s_F4_l01_a10_2)/2

s_F16_l01 = multinorm(F16_l01_1, F16_l01_2, F16_l01_3, F16_l01_4, F16_l01_5, steps1, a2)

#%%
plt.cla()

plt.plot(times, s_F4_l1000_a10, 'k-')
plt.plot(times, s_F4_l10_a10, 'r-')
plt.plot(times, s_F4_l1_a10, 'c-')
plt.plot(times, s_F4_l01_a10, 'g-')


plt.plot(times, s_F16_l1000, 'k.')
plt.plot(times, s_F16_l10, 'r.')
plt.plot(times, s_F16_l1, 'c.')
plt.plot(times, s_F16_l01, 'g.')

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

#%%
plt.cla()

plt.plot(times, s_F4_l1000_a10, 'k', label = '$\ell = 1000$')
plt.plot(times, s_F4_l10_a10, 'r', label = '$\ell = 10$')
plt.plot(times, s_F4_l1_a10, 'c', label = '$\ell = 1$')
plt.plot(times, s_F4_l01_a10, 'g', label = '$\ell = 0.1$')

plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)

plt.yticks([0.2, 0.4, 0.6, 0.8, 1])

plt.legend(fontsize = 14)

plt.savefig('4_a10.png', bbox_inches = 'tight', dpi = 300)

#%%
plt.cla()

plt.plot(times, s_F16_l1000, 'k', label = '$\ell = 1000$')
plt.plot(times, s_F16_l10, 'r', label = '$\ell = 10$')
plt.plot(times, s_F16_l1, 'c', label = '$\ell = 1$')
plt.plot(times, s_F16_l01, 'g', label = '$\ell = 0.1$')

plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)

plt.yticks([0.2, 0.4, 0.6, 0.8, 1])

plt.legend(fontsize = 14)

plt.savefig('16_a20.png', bbox_inches = 'tight', dpi = 300)

#%%
plt.cla()
plt.plot(times, s_F4_l1000, 'c', label = '$\phi = 0.79%$')
plt.plot(times, s_F8_l1000, 'g', label = '$\phi = 1.57%$')
plt.plot(times, s_F16_l1000, 'r', label = '$\phi = 3.14%$')
plt.legend(fontsize = 14)
plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)
plt.savefig('SP4v8v16.png', bbox_inches = 'tight', dpi = 300)

#%% Same particle density, diff box size
plt.cla()
plt.plot(times, s_F4_l1000_10x10, 'c', label = '10x10 box')
plt.plot(times, s_F16_l1000, 'g', label = '20x20 box')
plt.legend(fontsize = 14)
plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)
plt.savefig('SP4v16.png', bbox_inches = 'tight', dpi = 300)

plt.cla()
plt.clf()
plt.plot(times, s_F4_l1000, 'c', label = '4 particles')
plt.plot(times, s_F8_l1000, 'g', label = '8 particles')
plt.plot(times, s_F16_l1000, 'k', label = '16 particles')
plt.legend(fontsize = 14)
plt.ylabel(r'$\Vert s \Vert$', fontsize = 14)
plt.xlabel(r'$\bar{t}$', fontsize = 14)
plt.savefig('SP4v8v16.png', bbox_inches = 'tight', dpi = 300)

#%% F CM Plotting
f = 14
s = 50

fig, ax = plt.subplots(figsize = (8,6))

plt.cla()

# plt.plot(times, s_F, '-r', label = 'Deep Subphase')
# plt.plot(times[::s], s_F4_avg[::s], color = 'r', marker = "^", label = '$\phi$  = 0.7%')
# plt.plot(times[::s], s_F8_avg[::s], color = 'c', marker = "^", label = '$\phi$ = 1.57%')
# plt.plot(times[::s], s_F16_avg[::s], color = 'b', marker = "^", label = '$\phi$  = 3.14%')
# # plt.plot(times, s_CM, '-b', label = 'Confined Subphase')
plt.plot(times[::s], s_CM4_avg[::s], color = 'r', marker = "o", markerfacecolor = 'white', label = '$\phi$  = 0.7%')
plt.plot(times[::s], s_CM8_avg[::s], color = 'c', marker = "o", markerfacecolor = 'white', label = '$\phi$ = 1.57%')
plt.plot(times[::s], s_CM16[::s], 'b', marker = "o", markerfacecolor = 'white', label = '$\phi$  = 3.14%')

red_patch = mpatches.Patch(color = 'red', label = '$\phi$  = 0.7%')
green_patch = mpatches.Patch(color = 'green', label = '$\phi$  = 1.57%')
blue_patch = mpatches.Patch(color = 'blue', label = '$\phi$  = 3.14')

plus_patch = mlines.Line2D([], [], color = 'k', marker='^', linestyle='None',
                          markersize=10, label='Deep subphase')
o_patch = mlines.Line2D([], [], color = 'k', marker='o', linestyle='None',
                          markersize=10, markerfacecolor = 'white', label='Confined subphase')
# red_patch = mlines.Line2D([], [], color = 'r', marker='_', linestyle='None',
#                           markersize=10, label='$\phi$  = 0.7%')
# green_patch = mlines.Line2D([], [], color = 'c', marker='_', linestyle='None',
#                           markersize=10, label='$\phi$  = 1.57%')
# blue_patch = mlines.Line2D([], [], color = 'b', marker='_', linestyle='None',
#                           markersize=20, label='$\phi$  = 3.14%')
# plus_patch = mpatches.Patch(marker = "^", label = 'Deep subphase')
# o_patch = mpatches.Patch(marker = "o", label = 'Confined subphase')

plt.yticks([0.9,0.95,1], fontsize = f+2)
# plt.yticks([0.4, 0.6, 0.8, 1], fontsize = f)
plt.xticks([0,200,400,600,800,1000], fontsize = f+2)

# plt.legend(handles=[plus_patch, o_patch, red_patch, green_patch, blue_patch], fontsize = f)
plt.legend(fontsize = f, loc = 'lower left')

plt.xlabel(r'$t$', fontsize = f + 3)
plt.ylabel(r'$\Vert s \Vert$', fontsize = f + 3)

plt.title('Mixing norm, shallow subphase', fontsize = f + 3)
# plt.title('Mixing norm, confined membrane, $l_{sd}$ = 1.0', fontsize = f + 2)

plt.tight_layout()
plt.savefig('C,detailed.png', bbox_inches = 'tight', dpi = 300)
# ax.set_yscale('log')
plt.show()


#%%

fig, ax = plt.subplots(figsize = (8,6))

# plt.plot(times, s_F, '-r', label = 'Deep Subphase')
plt.plot(times[::s], s_F4_avg[::s], color = 'r', marker = "^", label = '4 Dipoles - Deep')
plt.plot(times[::s], s_F8_avg[::s], color = 'c', marker = "^", label = '8 Dipoles - Free')
plt.plot(times[::s], s_F16_avg[::s], color = 'b', marker = "^", label = '16 Dipoles - Free')
# plt.plot(times, s_CM, '-b', label = 'Confined Subphase')
# plt.errorbar(times[::s], s_CM4_avg[::s], yerr=stdev_CM4[::s], color = 'r', marker = "o", markerfacecolor = 'white', label = '4 Dipoles - Confined')
# plt.errorbar(times[::s], s_CM8_avg[::s], yerr=stdev_CM8[::s], color = 'c', marker = "o", markerfacecolor = 'white', label = '8 Dipoles - Confined')
# plt.plot(times[::s], s_CM16[::s], 'b', marker = "o", markerfacecolor = 'white', label = '16 Dipoles - Confined')

# red_patch = mpatches.Patch(color = 'red', label = '4 particles')
# green_patch = mpatches.Patch(color = 'green', label = '8 particles')
# blue_patch = mpatches.Patch(color = 'blue', label = '16 particles')

plus_patch = mlines.Line2D([], [], color = 'k', marker='^', linestyle='None',
                          markersize=10, label='Deep subphase')
o_patch = mlines.Line2D([], [], color = 'k', marker='o', linestyle='None',
                          markersize=10, markerfacecolor = 'white', label='Confined subphase')
red_patch = mlines.Line2D([], [], color = 'r', marker='_', linestyle='None',
                          markersize=10, label='4 particles')
green_patch = mlines.Line2D([], [], color = 'c', marker='_', linestyle='None',
                          markersize=10, label='8 particles')
blue_patch = mlines.Line2D([], [], color = 'b', marker='_', linestyle='None',
                          markersize=10, label='16 particles')
# plus_patch = mpatches.Patch(marker = "+", label = 'Deep subphase')
# o_patch = mpatches.Patch(marker = "o", label = 'Confined subphase')

plt.yticks([0.9,0.95,1], fontsize = f+2)
# plt.yticks([0.4, 0.6, 0.8, 1], fontsize = f)
plt.xticks([0,200,400,600,800,1000], fontsize = f+2)

plt.legend(handles=[plus_patch, o_patch, red_patch, green_patch, blue_patch], fontsize = f)
# plt.legend(fontsize = f)

plt.xlabel(r'$t$', fontsize = f + 3)
plt.ylabel(r'$\Vert s \Vert$', fontsize = f + 3)

plt.title('Mixing norm, deep vs confined', fontsize = f + 3)
# plt.title('Mixing norm, confined membrane, $l_{sd}$ = 1.0', fontsize = f + 2)

plt.tight_layout()
plt.savefig('FvsC,errors.png', bbox_inches = 'tight', dpi = 300)
# ax.set_yscale('log')
plt.show()

#%%
path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/0/Mixing'
pattern = '*.npy'
conc = glob.glob(f'{path}/{pattern}')
# %%
