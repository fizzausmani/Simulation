#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import seaborn as sns
from scipy.stats import poisson, chisquare

#%%
pos_f01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/FM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,30000]
pos_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd1 isotropic pos.npy')[:,:,30000]
pos_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic pos.npy')[:,:,30000]
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic ic.csv')

N = int(ic.loc[4][1])
# T = int(ic.loc[0][1]) 
# steps = int(ic.loc[2][1])

#%%
distances_f01 = np.zeros([N, N])
distances_01 = np.zeros([N, N])
distances_001 = np.zeros([N, N])

indices_f01 = np.zeros([N, N])
indices_01 = np.zeros([N, N])
indices_001 = np.zeros([N, N])

# for n in range(steps-3,steps):
posf01 = np.transpose(pos_f01[:,:])
pos01 = np.transpose(pos_01[:,:])
pos001 = np.transpose(pos_001[:,:])

nbrs_f01 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(posf01)
nbrs_01 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos01)
nbrs_001 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos001)

distances_f01[:,:], indices_f01[:,:] = nbrs_001.kneighbors(posf01)
distances_01[:,:], indices_01[:,:] = nbrs_01.kneighbors(pos01)
distances_001[:,:], indices_001[:,:] = nbrs_001.kneighbors(pos001)

indices_f01[:,:] = indices_f01[:,:]*(distances_f01[:,:] < 1.5)
indices_01[:,:] = indices_01[:,:]*(distances_01[:,:] < 1.5)
indices_001[:,:] = indices_001[:,:]*(distances_001[:,:] < 1.5)

#%%
counter_01 = np.zeros(N)
counter_001 = np.zeros(N)
counter_f01 = np.zeros(N)

# for n in range(0,steps):
for i in range(0,N):
    for j in range(0,N):
        counter_01[i] += 1*(indices_01[i,j] != 0)
        counter_001[i] += 1*(indices_001[i,j] != 0)
        counter_f01[i] += 1*(indices_f01[i,j] != 0)

# %% probability

def probability(counts):
    a = Counter(counts)
    p1 = a[1]/N
    p2 = a[2]/N
    p3 = a[3]/N
    p4 = a[4]/N
    p5 = a[5]/N
    
    prob = np.array([p1, p2, p3, p4, p5])
    
    return prob

probability_01 = probability(counter_01)
probability_001 = probability(counter_001)
probability_f01 = probability(counter_f01)

#%%
f = 14
x1 = np.array([1,2,3,4,5])
barWidth = 0.25
x2 = [x + barWidth for x in x1]
x3 = [x + barWidth for x in x2]
plt.cla()
plt.clf()

plt.bar(x1, probability_f01, hatch ="/", label = '$FM, \ell = 0.1$', color = 'black', width = barWidth)
plt.bar(x2, probability_01, edgecolor = 'k', hatch = "//", label = '$CM, \ell H = 0.1$', color = 'red', width = barWidth)
plt.bar(x3, probability_001, edgecolor = 'k', hatch = "//", label = '$CM, \ell H = 0.01$', color = 'coral', width = barWidth)
plt.legend(fontsize = f)
plt.xlabel('Number of neighbors, N', fontsize = f)
plt.ylabel('$p(N)$', fontsize = f)
plt.xticks([r + 1.25 for r in range(len(x1))], 
        ['1', '2', '3', '4', '5'])
plt.savefig('Nearest Neighbor Analysis, N625_T1500.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#%% poisson fit
mu = np.mean(counter_f01)  # Poisson parameter is the mean of the data
x = np.arange(poisson.ppf(0.001, mu), poisson.ppf(0.99, mu))
plt.cla()
plt.plot(x, poisson.pmf(x, mu), 'r-', label='poisson pmf')

# Plot your data
plt.hist(counter_f01, bins=x, density=True, alpha=0.6, color='g')

plt.legend()
plt.show()

#%%
from scipy.stats import gamma

# Calculate the observed frequencies
observed_freq, bins = np.histogram(counter_001, bins=20, density=False)

# Calculate the expected frequencies
a, loc, scale = gamma.fit(counter_001)
expected_freq = [gamma.pdf(i, a, loc, scale) * len(counter_001) for i in bins[:-1]]

# Perform the chi-square test
chi2, p = chisquare(observed_freq, f_exp=expected_freq)

print(f'Chi-square statistic: {chi2}')
print(f'p-value: {p}')