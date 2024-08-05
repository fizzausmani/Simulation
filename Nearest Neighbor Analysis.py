#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from scipy.stats import poisson, chisquare

#%%
pos_f01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/FM T10000 N625 lattice sd0.1 isotropic continued pos.npy')[:,:,-1]
pos_01 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd1 isotropic continued 2 pos.npy')[:,:,-1]
pos_001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.1 isotropic continued 4 pos.npy')[:,:,-1]
pos_0001 = np.load('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T10000 N625 lattice sd0.01 isotropic continued 5 pos.npy')[:,:,-1]
ic = pd.read_csv('/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Python/Stability Analysis/CM T5000 N625 lattice sd0.1 isotropic ic.csv')

N = int(ic.loc[4][1])

#%%
posf01 = np.transpose(pos_f01)
pos01 = np.transpose(pos_01)
pos001 = np.transpose(pos_001)
pos0001 = np.transpose(pos_0001)

nbrs_f01 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(posf01)
nbrs_01 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos01)
nbrs_001 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos001)
nbrs_0001 = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(pos0001)

distances_f01, indices_f01 = nbrs_f01.kneighbors(posf01)
distances_01, indices_01 = nbrs_01.kneighbors(pos01)
distances_001, indices_001 = nbrs_001.kneighbors(pos001)
distances_0001, indices_0001 = nbrs_0001.kneighbors(pos0001)

indices_f01 = indices_f01*(distances_f01 < 1.5)
indices_01 = indices_01*(distances_01 < 1.5)
indices_001 = indices_001*(distances_001 < 1.5)
indices_0001 = indices_0001*(distances_0001 < 1.5)

#%%
counter_f01 = np.zeros(N)
counter_01 = np.zeros(N)
counter_001 = np.zeros(N)
counter_0001 = np.zeros(N)

# for n in range(0,steps):
for i in range(0,N):
    for j in range(0,N):
        counter_f01[i] += 1*(indices_f01[i,j] != 0)
        counter_01[i] += 1*(indices_01[i,j] != 0)
        counter_001[i] += 1*(indices_001[i,j] != 0)
        counter_0001[i] += 1*(indices_0001[i,j] != 0)

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
probability_0001 = probability(counter_0001)
probability_f01 = probability(counter_f01)

#%%
f = 14
x1 = np.array([1,2,3,4,5])
barWidth = 0.20
x2 = [x + barWidth for x in x1]
x3 = [x + barWidth for x in x2]
x4 = [x + barWidth for x in x3]
plt.cla()
plt.clf()

plt.bar(x1, probability_f01, hatch ="/", label = '$FM, \ell = 0.1$', color = 'black', width = barWidth)
plt.bar(x2, probability_01, edgecolor = 'w', hatch = "//", label = '$CM, \ell H = 0.1$', color = 'maroon', width = barWidth)
plt.bar(x3, probability_001, edgecolor = 'k', hatch = "//", label = '$CM, \ell H = 0.01$', color = 'coral', width = barWidth)
plt.bar(x4, probability_0001, edgecolor = 'k', hatch = "//", label = '$CM, \ell H = 0.001$', color = 'sandybrown', width = barWidth)
plt.legend(fontsize = f-2, loc = 'upper right')
plt.xlabel('Number of neighbors, N', fontsize = f)
plt.xlim(0.5, 6)
plt.ylabel('$p(N)$', fontsize = f)
plt.xticks([r + 1.25 for r in range(len(x1))], 
        ['1', '2', '3', '4', '5'])
save_path = '/Users/fizzausmani/Library/CloudStorage/Box-Box/Research/Updates/Summer 2024/'
# plt.savefig(save_path + 'Nearest Neighbor Analysis, N625.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#%%
def counts(counts):
    a = Counter(counts)
    p1 = a[1]
    p2 = a[2]
    p3 = a[3]
    p4 = a[4]
    p5 = a[5]
    
    return np.array([p1, p2, p3, p4, p5])

fitter_input_f01 = counts(counter_f01)
fitter_input_01 = counts(counter_01)
fitter_input_001 = counts(counter_001)
fitter_input_0001 = counts(counter_0001)

#%% poisson fit
mu = np.mean(counter_001)  # Poisson parameter is the mean of the data
x = np.arange(poisson.ppf(0.1, mu), poisson.ppf(0.9, mu))
plt.cla()
plt.plot(x, poisson.pmf(x, mu, 1), 'r-', label='poisson pmf')

# Plot your data
plt.hist(counter_001, bins=5, density=True, alpha=0.6, color='g')

plt.legend()
plt.show()

# Calculate the observed frequencies
observed_freq, bins = np.histogram(counter_001, bins=20, density=False)

# Calculate the expected frequencies
expected_freq = [poisson.pmf(i, mu) * len(counter_001) for i in range(len(bins)-1)]

# Perform the chi-square test
chi2, p_value = chisquare(observed_freq, f_exp=expected_freq)

print(f'Chi-square statistic: {chi2}')
print(f'p-value: {p_value}') 

prob = poisson.cdf(x, mu)
np.allclose(x, poisson.ppf(prob, mu))
# %%
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions

f = Fitter((counter_f01),
           distributions=["gamma",
                          "poisson",
                          "geom"])
f.fit()
f.summary()

#%%
f = Fitter((counter_01),
           distributions=["gamma",
                          "poisson",
                          "geom"])
f.fit()
f.summary()

#%%
f = Fitter((counter_001),
           distributions=["gamma",
                          "poisson",
                          "geom"])
f.fit()
f.summary()
#%%

f = Fitter((counter_0001),
           distributions=["gamma",
                          "poisson",
                          "geom"])
f.fit()
f.summary()
# %%
