#%%
import os
import sys
import scipy.io as sio  # load save matlab files
import numpy as np  # np matrix functions
import scipy.signal as sg
import scipy.interpolate as interpolate
import pickle
import scipy.spatial.distance as dis
from scipy.signal import stft
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.rc('font', size=14) # default text size
#%%

data = loadmat('Data/CW_DB.mat', mdict=None, appendmat=True)
NAMES = pickle.load( open( "data_west_order.p", "rb" ) )
x_data_names = NAMES['x_names']
fr_data_names = NAMES['fr_names']
fs_data = NAMES['fs']

#NAMES = pickle.load( open( "drive/My Drive/Detección_de_fallas/data_west_order.p", "rb" ) )

x_data_names
#%%
types = []
for i in x_data_names:
    tmp = int(i[1:4])
    if tmp>=282:
        types.append('FE')
    else:
        types.append('DE')

#%%

fault_type = []
for i in x_data_names:
    tmp = int(i[1:4])
    if (tmp>=56 and tmp<=59) or tmp == 173:
        fault_type.append('BPFI')
    elif (tmp>=118 and tmp<=125) or (tmp>=224 and tmp<=229) or (tmp==192) or (tmp>=282 and tmp<=293):
        fault_type.append('BSF')
    else:
        fault_type.append('BPFO')
#%%

fault_freq = {'BPFODE':3.585,'BPFOFE':3.053,'BPFIDE':5.415,'BPFIFE':4.947,'BSFDE':2.357,'BSFFE':1.994,'FTFDE':0.3983,'FTFFE':0.3816}
#%%

for i in range(len(x_data_names)):
    x = data[x_data_names[i]].ravel()
    fr = data[fr_data_names[i]][0, 0] / 60
    fs = fs_data[i]

    x = x - np.mean(x)
    mfreq = .05 * fr  # fr rotational speed
    Nw = int(2 ** np.fix(np.log2(1 / mfreq * fs)))  # window for computation of SK
    Nfft = 2 * Nw
    Noverlap = round(3 / 4 * Nw)

    ent = pickle.load(open("Results_w/Results_W2/" + x_data_names[i] + "_w.p", "rb"))
    f = ent['f']*fs/fr
    psd = ent['psd']

    name_freq = fault_type[i] + types[i]

    Bfq = fault_freq[name_freq]
    print(Bfq)

    psd = psd / max(psd)
    psd_max = 1.1
    # psd_max = 0.1

    plt.figure()
    plt.plot(f, psd)
    plt.xlim([0, 5.5 * Bfq])
    plt.ylim([0, psd_max])
    plt.xlabel('Order [xn]')
    plt.ylabel('PSD')
    xcoords = (np.r_[:10] * Bfq) + Bfq
    for xc in xcoords:
        plt.axvline(x=xc, color='red', linestyle=':')
    plt.savefig('Figures/' + x_data_names[i] + '_entropy_w.pdf', bbox_inches='tight')
    # print('ok data')
