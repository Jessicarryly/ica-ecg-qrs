print(__doc__)

import matplotlib.pyplot as plt
import scipy.io
from sklearn.decomposition import FastICA, PCA

signals = scipy.io.loadmat('signalsByQRS1.mat')
sig = signals['signalPQRS'];
#print signals['signalPQRS'][0][0:101]

#import matplotlib.pyplot as plt
plt.figure()
plt.plot(sig[50026][0:100])

# Compute ICA AF
ica = FastICA()
X = sig[0:50025]
S_AF = ica.fit_transform(X)  # Reconstruct signals
A_AF = ica.mixing_  # Get estimated mixing matrix
print len(A_AF)

# For comparison, compute PCA
pca = PCA()
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

###############################################################################
# Plot results
plt.figure()

plt.plot(S_AF[0],color='red')
plt.plot(S_AF[1],color='steelblue')
plt.plot(S_AF[2],color='orange')
plt.plot(S_AF[3],color='green')



# Compute ICA NORMAL
ica = FastICA() #n_components=7
S_N = ica.fit_transform(sig[50026:len(sig)-1])  # Reconstruct signals
A_N = ica.mixing_  # Get estimated mixing matrix
print len(A_N)

plt.figure()
plt.plot(S_N[0],color='red')
plt.plot(S_N[1],color='steelblue')
plt.plot(S_N[2],color='orange')
plt.plot(S_N[3],color='green')
plt.show()

# plot results
plt.figure()

models = [X,  X, S_AF, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()


# models = [S_[0], S_[1], S_[2], S_[3]]
# names = ['IC1',
#          'IC2',
#          'IC3',
#          'IC4']
# colors = ['red', 'steelblue', 'orange', 'blue']
#
# for ii, (model, name) in enumerate(zip(models, names), 1):
#     plt.subplot(4, 1, ii)
#     plt.title(name)
#     for sig, color in zip(model, colors):
#         plt.plot(sig, color=color)
#
# plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)