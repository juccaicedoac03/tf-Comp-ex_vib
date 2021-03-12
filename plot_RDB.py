
import os
import sys
import scipy.io as sio  # load save matlab files
import numpy as np  # np matrix functions
import scipy.signal as sg
import scipy.interpolate as interpolate
import pickle
import scipy.spatial.distance as dis
from scipy.signal import stft
import matplotlib.pyplot as plt
from functions.IAS_est import printxtipsusr
#%%

data = sio.loadmat('Dataz/D10_SK_angle_domain.mat')
datas = data['datas'].ravel()
SampPerRev = data['SampPerRev'][0,0]

fs = SampPerRev

Nw       = 2**np.fix(np.log2(1/.4*SampPerRev))*2**3
Nw       = int(Nw)
Nfft     = 2*Nw
Noverlap = round(3/4*Nw)
#%%

ent = pickle.load( open( "Results/RDB.p", "rb" ) )
entropy = ent['entropy']
#%%

E = np.concatenate((entropy,entropy[::-1]),axis=0)
b = np.fft.fftshift(np.real(np.fft.ifft(E)))
datas = sg.fftconvolve(datas, b, mode='same')
#%%

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datae)**2
datae = datae-np.mean(datae)
sp = np.abs(np.fft.fft(datae,Nfft)/Nfft)**2
freq = np.fft.fftfreq(sp.shape[-1])

sp = sp[:int(len(sp)/2)]
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev
#%%

fxh =62/61
freq = freq/fxh

fig1 = plt.figure()
plt.plot(freq, sp)
plt.axis([freq[0],100,0,1.1*max(sp[(freq>7)*(freq<100)])])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

Bfq=7.69
xcoords = np.r_[:12*Bfq:Bfq]+Bfq
plt.vlines(x=xcoords,ymin=0, ymax=1.5*max(sp), color='red',label= 'BPFO',linestyle='dashdot')
plt.legend(loc='upper right', shadow=True)
plt.savefig('Figures/aircraft_00_entropy_.pdf', bbox_inches='tight')

#%% datatips zoom
f = freq

fig2 = plt.figure()
plt.plot(f,sp)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

# finding max around 7.7
fxh  = 7.7
dfxh = 2e-1
px = max(sp[(f>fxh-dfxh)*(f<fxh+dfxh)])
fx = f[sp==px]

K = 1
fcage = 0#0.54527385
xtip = np.zeros(K)
xtip[0] = fx-fcage
for k in np.r_[:K]-2:
    xtip[k+2],_ =printxtipsusr(f,sp,xheigh=xtip[k+2],deltapsd=2e-2,tips='x',prt=1)
    try:
        xtip[k+3] = xtip[k+2]+fcage/2
    except IndexError:
        print('Are u finish?')
pxtip = int((len(xtip)-1)/2)
plt.axis([xtip[pxtip]-2,xtip[pxtip]+2,0,1.1*max(sp[(freq>7)*(freq<9)])])

plt.legend([r"$\left|X(\Theta)\right|^2$"])
plt.show()
plt.savefig('Figures/aircraft_zoom_entropy.pdf',bbox_inches='tight')
