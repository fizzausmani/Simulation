#%%
import numpy as np
import matplotlib.pyplot as plt

pos_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd1 isotropic continued pos.npy')[:,:,-1]
p_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd1 isotropic continued p.npy')[:,:,-1]
pos_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued 3 pos.npy')[:,:,-1]
p_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued 3 p.npy')[:,:,-1]
pos_0001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 3 pos.npy')[:,:,-1]
p_0001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 3 p.npy')[:,:,-1]

def pair_coords(pos,p):
    N = pos.shape[1]
    rfx=[]
    rfy=[]

    for i in range(0, N):
        origin = (pos[:, 0]) 
        origin = origin[:, np.newaxis]* np.ones([N])
        p_origin = (p[:, 0])
        p_origin = p_origin[:, np.newaxis] * np.ones([N])
        r = pos - origin
        x = np.dot(np.transpose(r), p_origin)
        x = x[:, 0]  
        rdotpp = p_origin*x
        y_vec = r - rdotpp
        # check = np.dot(np.transpose(rabs), p_origin)   
        y = np.sqrt(y_vec[0, :]**2 + y_vec[1, :]**2)
        rfx.append(np.abs(x))
        rfy.append(y)
        pos = np.roll(pos, 1, axis=1)
        p = np.roll(p, 1, axis=1)

    rfx = np.array(rfx)
    rfy = np.array(rfy)
    return rfx, rfy

def binning(x,y):
    x_reshaped = np.ravel(x)
    y_reshaped = np.ravel(y)

    binedges_x = np.linspace(0, 4, 101)
    binedges_y = np.linspace(0, 4, 101) 

    x_indices = (x_reshaped//(binedges_x[1] - binedges_x[0]))
    y_indices = (y_reshaped//(binedges_y[1] - binedges_y[0]))

    rf_coords = np.array([x_reshaped, y_reshaped])
    a = [rf_coords[0,:] < 4]
    b = [rf_coords[1,:] < 4]
    rf_coords = rf_coords*a*b
    mask = (rf_coords != 0).all(axis=0)
    rf_coords_filtered = rf_coords[:,mask]

    x_indices = (rf_coords_filtered[0,:]//(binedges_x[1] - binedges_x[0]))
    y_indices = (rf_coords_filtered[1,:]//(binedges_y[1] - binedges_y[0]))

    bin_counts = np.zeros([len(binedges_x), len(binedges_y)])
    for i in range(0,x_indices.shape[0]):
        bin_counts[int(x_indices[i]), int(y_indices[i])] += 1
    # bin_counts = bin_counts/np.sum(bin_counts)

    return bin_counts

def plotting(bin_counts, sd):
    x = np.linspace(0, 4, bin_counts.shape[0])
    y = np.linspace(0, 4, bin_counts.shape[0]) 

    plt.cla()
    plt.clf()

    contour = plt.contourf(x, y, bin_counts.T, levels = 500, cmap = 'magma_r')
    # plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(contour)
    cbar_ticks = np.int64(np.linspace(0, np.max(bin_counts), 9))
    cbar.set_ticks(cbar_ticks)
    
    save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Updates/Summer 2024/'
    plt.savefig(save_path + 'Pair Interactions sd' + str(sd) + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

x_01, y_01 = pair_coords(pos_01, p_01)
x_001, y_001 = pair_coords(pos_001, p_001)
x_0001, y_0001 = pair_coords(pos_0001, p_0001)

bin_01 = binning(x_01, y_01)
bin_001 = binning(x_001, y_001)
bin_0001 = binning(x_0001, y_0001)

plotting(bin_01, 1)
plotting(bin_001, 0.1)
plotting(bin_0001, 0.01)
