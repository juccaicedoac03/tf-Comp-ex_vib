import os # change dir
from os import listdir
from os.path import isfile, join
import sys 
Wdir= os.getcwd()
sys.path.append(Wdir+'\functions_stsk')# add to path
dbdir = 'G:/Mon Drive/GCPDS Database/Case_western_N1N2'
import numpy as np
import scipy.io as sio # load save matlab files
import scipy.signal as sg # signal pack
import matplotlib.pyplot as plt # plotting
from scipy.io import wavfile # read wav files
from matplotlib.gridspec import GridSpec # easy subplots

from functions.STSK import SK_W,PSD_envW,\
printxtips,printxtipsusr
plt.rc('font', size=14) # default text size
#%% loading data
#dbdir += '/N1N2'
dbdir += '/Y1'
onlyfiles = [f for f in listdir(dbdir) if isfile(join(dbdir, f)) and 'mat' in f]
data = {'Database':None}
for f in onlyfiles:
    mat_contents = sio.loadmat(dbdir+'/'+f, mdict=None, appendmat=True)
    del mat_contents['__globals__']
    del mat_contents['__header__']
    del mat_contents['__version__']
    for k,v in mat_contents.items():#keys,values
        data[k.lower()] = v
del mat_contents,k,v,f,onlyfiles
del data['Database']
#locals().update(data) # to load all files to local

#%% SK filtering proposal
#fs = 12e3
#x = data['x059_de_time']
#data['x059rpm'] = [1730]
#x = x-np.mean(x)
#fr = data['x059rpm'][0]/60
#sio.savemat('data/CW_DB.mat', 
#                data,do_compression=True)
data = sio.loadmat('data/CW_DB.mat', mdict=None, appendmat=True)
#%%
plt.close('all')
fs = 12e3
recordstr = '197_fe'#de/fe/ba
x = data['x'+recordstr+'_time']
fr = data['x'+recordstr[:-3]+'rpm'][0]/60
#fs = 48e3
#x = data['x239_de_time']
#fr = data['x239rpm'][0]/60
x = x-np.mean(x)

mfreq = 0.4*fr # fr rotational speed
Nw1      = 2**np.ceil(np.log2(1/mfreq*fs))*2**3 # greater than Nw2 window for envelope spectrum
Nw1      = int(Nw1)
Nw2      = 2**8 # window for computation of SK
Nw2      = int(Nw2)
Nfft     = 2*Nw1
Nfft2    = 2*Nw2
Noverlap = round(3/4*Nw1)
Window   = np.kaiser(Nw1,beta=0) # beta 0 rectangular,5	Similar to a Hamming
# 6	Similar to a Hanning, 8.6	Similar to a Blackman

#%
filterr  = 1 # filtering with SK
psd,f,K,SK_w  = PSD_envW(x,Nfft,Noverlap,Window,Nw2,Nfft2,filterr)
f        = f*fs/fr
#%% plotting
fault_freq = {'BPFODE':3.585,'BPFOFE':3.053,'BPFIDE':5.415,'BPFIFE':4.947,\
              'BSFDE':2.357,'BSFFE':1.994,'FTFDE':0.3983,'FTFFE':0.3816}

# correction arround 1
#Bfq = 1
#dfxh = 1e-1
#px = max(psd[(f>Bfq-dfxh)*(f<Bfq+dfxh)])
#frc = f[psd==px]
#f   = f/frc

Bfq = fault_freq['BPFODE']
#Bfq = 3.1
#dfxh = 1e-1
#px = max(psd[(f>Bfq-dfxh)*(f<Bfq+dfxh)])
#Bfq = f[psd==px]


psd = psd/max(psd)
psd_max = 1.1
#psd_max = 0.1

plt.figure()
plt.plot(f,psd)
plt.xlim([0,5.5*Bfq])
plt.ylim([0,psd_max])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
xcoords = (np.r_[:10]*Bfq)+Bfq
for xc in xcoords:
    plt.axvline(x=xc,color='red',linestyle=':')
plt.show()
#plt.savefig('figures/'+recordstr+'.pdf',bbox_inches='tight')
recordstr
#plt.figure()
#plt.plot(f/fr,psd)
#plt.xlim([0,4.5*Bfq])
#plt.ylim([0,psd_max])
#plt.xlabel('Order [xn]')
#plt.ylabel('PSD')
#printxtips(f/fr,psd,Bfq,deltapsd,K=4)

plt.figure()
plt.plot(SK_w[:,1])