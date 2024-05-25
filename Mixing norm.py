#%% Imports
import glob as glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#%% Importing concentration data
path_f1000 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/Mixing/sd1000'
path_f10 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/Mixing/sd10'
path_f1 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/Mixing/sd1'
path_f01 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP8/Mixing/fsd01'
# path_c01 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/csd01'
# path_c1 = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Full Results/SP4/Mixing/csd1'

preamble = 'AD T1000 dt0.05 SP8 FM conc'
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
def norm(conc_path, steps, a, z):
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
s_l1000 = norm(c_f1000, steps1, a2, z2)
s_l10 = norm(c_f10, steps1, a2, z2)
s_l1 = norm(c_f1, steps1, a2, z2)
s_l01 = norm(c_f01, steps1, a2, z2)
# s_l01 = norm(c_c01, steps1, a1, z2)

#%%
plt.cla()
plt.plot(times, s_l1000, 'k')
plt.plot(times, s_l10, 'r')
plt.plot(times, s_l1, 'salmon')
plt.plot(times, s_l01, 'coral')
# plt.plot(times, s_01, 'b_')

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

savepath = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Updates/Spring 2024/05.30.2024/'

plt.legend(handles=[patch01, patch1, patch10, patch1000], fontsize = 14)
plt.savefig(savepath + 'N8.png', bbox_inches = 'tight', dpi = 300)
plt.show()
# %%
