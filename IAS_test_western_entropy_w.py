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

def PSD_envW(x,nfft,Noverlap,Window,Nwind2,nfft2,filterr):
    Window = Window.reshape(-1,1)[:,0]
    n = len(x)		# Number of data points
    nwind = len(Window) # length of window
    x = x.reshape(-1,1)[:,0]
    K = int(np.fix((n-Noverlap)/(nwind-Noverlap))) # Number of windows
    Noverlap2 = int(np.round(3/4*Nwind2))
    Window2 = np.hanning(Nwind2)
    
    # compute 
    index = np.r_[:nwind]
    #t = np.r_[:n]
    psd = np.zeros(nfft)
    SK_w = np.zeros([nfft2,K])
    for i in np.r_[:K]:
        xw = Window*x[index]
    	# filtering
        if filterr == 1:
            f,t,Z = stft(xw.ravel(),fs,nperseg=Noverlap2,nfft=nfft2)
            Xz = np.abs(Z)
            entropy = np.asarray([bayes_entropy2(Xz[i,:].reshape(1,-1),np.mean(np.std(Xz,axis=1)),3) for i in range(Xz.shape[0])])
            SK = np.concatenate((entropy,entropy[::-1]),axis=0)
            #SK,_,_,_ = SK_W(xw,nfft2,Noverlap2,Window2)
            SK_w[:,i] = SK.ravel()
            b = np.fft.fftshift(np.real(np.fft.ifft(SK)))
            xw = sg.fftconvolve(xw,b,mode='same') #xw = fftfilt(b,xw);
        xw = np.abs(sg.hilbert(xw)) # envelope
        xw = xw**2
        xw = xw - np.mean(xw)
        xw = sg.hilbert(xw)
        Xw = np.fft.fft(xw,nfft)/nfft
        psd = np.abs(Xw)**2 + psd
        index = index + (nwind - Noverlap)
    # normalize
    KMU = K*np.linalg.norm(Window)**2;	#Normalizing scale factor ==> asymptotically unbiased
    psd = psd/KMU
    
    freq = np.fft.fftfreq(psd.shape[-1])
    psd = psd[:int(len(psd)/2)]
    freq = freq[:int(len(freq)/2)]
    return psd,freq,K,SK_w


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
    filterr  = 1 # filtering with SK
    mfreq = 0.5*fr # fr rotational speed
    Nw1      = int(2**np.fix(np.log2(1/mfreq*fs))*2**3) # greater than Nw2 window for envelope spectrum
    Nw2      = int(2**np.fix(np.log2(1/mfreq*fs))) # window for computation of SK
    Nfft     = 2*Nw1
    Nfft2    = 2*Nw2
    Noverlap = round(3/4*Nw1)
    Window   = np.kaiser(Nw1,beta=0) # beta 0 rectangular,5	Similar to a Hamming


    psd,f,K,E_w = PSD_envW(x,Nfft,Noverlap,Window,Nw2,Nfft2,filterr)

    results = {}
    results['entropy_w'] = E_w
    results['K'] = K
    results['f'] = f
    results['psd'] = psd

    with open('Results_W/'+x_data_names[i]+'_w.p','wb') as handle:
        pickle.dump(results,handle)
    






