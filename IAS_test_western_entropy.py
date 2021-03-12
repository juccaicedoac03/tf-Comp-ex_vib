import os
import sys
import scipy.io as sio  # load save matlab files
import numpy as np  # np matrix functions
import scipy.signal as sg
import scipy.interpolate as interpolate
import pickle
import scipy.spatial.distance as dis
from scipy.signal import stft


def bayes_entropy2(X,r,tau):
    pos = 0
    Xm = []
    while pos+tau+1<X.shape[1]:
        Xm.append(X[:,pos:pos+tau])
        pos = pos+tau
    D = []
    for sim in Xm:
        if len(D)==0:
            D.append(sim.ravel())
        elif np.sum(dis.cdist(np.asarray(D),sim)<r)==False:
            D.append(sim.ravel())
    Xm = [i.ravel() for i in Xm]
    xx = np.asarray(Xm)
    d = dis.cdist(np.asarray(D),np.asarray(Xm))
    sig = np.median(d)
    simil = np.exp(-((d)**2)/(2*sig**2))
    #print(sum(simil.ravel()))
    Pd = np.mean(simil,axis=1)
    m  = np.mean(np.asarray([xx[:,i].reshape(xx[:,i].shape[0],1)*simil.T for i in range(xx.shape[1])]),axis=1).T
    var = np.mean(((dis.cdist(xx,m).T)**2)*simil,axis=1)
    d2 = dis.cdist(xx,m)
    simil2 = np.exp(-((d2)**2)/(2*var))
    Psd = np.mean(simil2,axis=0)
    Pds = Psd*Pd
    #print(len(D),len(Xm))
    E = -np.sum(Pds*np.log(Pds))/len(Xm)
    return E


data = sio.loadmat('Data/CW_DB.mat', mdict=None, appendmat=True)
NAMES = pickle.load( open( "Data/data_west_order.p", "rb" ) )

x_data_names = NAMES['x_names']
fr_data_names = NAMES['fr_names']
fs_data = NAMES['fs']

for i in range(len(x_data_names)):
    print('Test: '+str(i)+' of '+str(len(x_data_names)))
    x = data[x_data_names[i]]
    fr = data[fr_data_names[i]][0,0]/60
    fs = fs_data[i]

    x = x-np.mean(x)
    mfreq = .05*fr # fr rotational speed
    Nw = int(2**np.fix(np.log2(1/mfreq*fs))) # window for computation of SK
    Nfft = 2*Nw
    Noverlap = round(3/4*Nw)

    f,t,Z = stft(x.ravel(),fs,nperseg=Nw,nfft=Nfft)
    Xz = np.abs(Z)
    entropy = np.asarray([bayes_entropy2(Xz[i,:].reshape(1,-1),np.mean(np.std(Xz,axis=1)),3) for i in range(Xz.shape[0])])

    results = {}
    results['entropy'] = entropy

    with open('Results/'+x_data_names[i]+'.p','wb') as handle:
        pickle.dump(results,handle)
    






