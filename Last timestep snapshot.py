#%%
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

pos_f01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/FM T5000 N625 lattice sd0.1 isotropic continued pos.npy')[:,:,-1]
p_f01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/FM T5000 N625 lattice sd0.1 isotropic continued p.npy')[:,:,-1]
pos_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd1 isotropic continued pos.npy')[:,:,-1]
p_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd1 isotropic continued p.npy')[:,:,-1]
pos_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued 3 pos.npy')[:,:,-1]
p_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued 3 p.npy')[:,:,-1]
pos_0001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 3 pos.npy')[:,:,-1]
p_0001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 3 p.npy')[:,:,-1]

N = pos_01.shape[1]
a = 60

#%%
def laststep(pos, p, a, sd: any):
    fig, ax = plt.subplots(figsize=(7,6))
    plt.cla()

    ax.set_aspect(1)
    
    for i in range(N):
        ax.set_aspect(1)
        circle = plt.Circle((pos[0,i], pos[1,i]), 0.5, edgecolor = 'r',\
                            facecolor = 'Gainsboro')
        ax.add_artist(circle)
        
        ax.arrow(pos[0,i], pos[1,i], p[0,i]/2, p[1,i]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')
        ax.arrow(pos[0,i], pos[1,i], -p[0,i]/2, -p[1,i]/2, head_width=0.1,\
                  head_length=0.1, fc='k', ec='k')
    
    ax.axis([-a/2, a/2, -a/2, a/2])
    ax.tick_params(top = True, labeltop = True, right = True, labelright = True)

    plt.savefig(save_path + 'Last step sd' + str(sd) + '.png', dpi = 300, bbox_inches = 'tight')

#%%
N = pos_01.shape[1]
a = 60
save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Updates/Summer 2024/'

laststep(pos_01, p_01, a, 'c01')
laststep(pos_001, p_001, a, 'c001')
laststep(pos_0001, p_0001, a,'c0001')
laststep(pos_f01, p_f01, a, 'f01')