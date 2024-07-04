#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
pos = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0l1 isotropic pos.npy')[:,:,-1]
p = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic p.npy')[:,:,-1]

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

rfx_reshaped = np.ravel(rfx)
rfy_reshaped = np.ravel(rfy)
plt.scatter(rfy_reshaped, rfx_reshaped, s = 0.5, alpha = 0.5)
plt.xlim(0,4)
plt.ylim(0,4)

# Assuming rfx_reshaped and rfy_reshaped are defined as in the excerpt

# 1. Define the bin edges
bin_edges_x = np.linspace(1.5, 4, num=200)  # 20 bins from 0 to 4 for x
bin_edges_y = np.linspace(1.5, 4, num=200)  # 20 bins from 0 to 4 for y

# 2. Digitize the data
# Adjust bin_indices_x and bin_indices_y to ensure they are within valid range
bin_indices_x = np.digitize(rfx_reshaped, bin_edges_x)  # Convert to 0-based indices
bin_indices_y = np.digitize(rfy_reshaped, bin_edges_y)

bin_indices_x = np.minimum(bin_indices_x, len(bin_edges_x) - 1)  # Subtract 2 because of 0-based indexing and to stay within bounds
bin_indices_y = np.minimum(bin_indices_y, len(bin_edges_y) - 1)

# Continue with aggregation and visualization as before
bin_counts = np.zeros((len(bin_edges_x), len(bin_edges_y)))
for x_idx, y_idx in zip(bin_indices_x, bin_indices_y):
    bin_counts[x_idx, y_idx] += 1

hist, xbins, ybins = np.histogram2d(x=rfx_reshaped, y=rfy_reshaped, bins=(np.arange(0, 4, 0.02), np.arange(0, 4, 0.02)))

X, Y = np.meshgrid((xbins[:-1] + xbins[1:]) / 2, (ybins[:-1] + ybins[1:]) / 2)
fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contour(X, Y, hist.T, levels=500)
# ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f")
vmin = 0.0
vmax = 15
levels = np.linspace(vmin, vmax, 200)
contourf = ax.contourf(X, Y, hist.T, levels=levels, cmap='magma_r')  # Using 100 levels and a colormap for better visualization
plt.colorbar(contourf, ax=ax)  # Add a colorbar to the plot
ax.set_title('Contour Plot of Pair Interactions')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Updates/Summer 2024/'
plt.savefig(save_path + 'Pair Interactions sd01.png', bbox_inches='tight', dpi=300)
plt.show()
