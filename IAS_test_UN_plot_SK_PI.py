import os
import sys

Wdir = os.getcwd()
sys.path.append(Wdir + '\functions')  # add to path
import scipy.io as sio  # load save matlab files

import numpy as np  # np matrix functions
import scipy.signal as sg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt  # plotting
from matplotlib.gridspec import GridSpec  # easy subplots
from functions.IAS_est import f_est_linear, iaslinearapproxv2
from functions.STSK import COT_intp2, PSD_envW, SK_W

plt.rc('font', size=14)  # default text size
# %% loading data
filed = 'Data/Test_16_PI_STASK'
mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)

fs = int(mat_contents['fs_ori'])
data = mat_contents['data_save'][:, 1]
f_est = mat_contents['f_est_save'][:, 1]
# %% SK - angle resampled - baseline 00
# % SK filtering Jerome
Nw = int(2 ** np.fix(np.log2(1 / 0.39 * fs)))
Nw = int(Nw)
Nfft = 2 * Nw
Noverlap = round(3 / 4 * Nw)

SK, _, _, _ = SK_W(data, Nfft, Noverlap, Nw)
b = np.fft.fftshift(np.real(np.fft.ifft(SK)))
datas = sg.fftconvolve(data, b, mode='same')

# % angular resampling
SampPerRev = int(2 ** np.fix(np.log2(fs / 2 / max(f_est))) * 2 ** 2)
datas, tc, SampPerRev, _ = COT_intp2(f_est, SampPerRev, datas, fs)

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datas) ** 2
datae = datae - np.mean(datae)
sp = np.abs(np.fft.fft(datae, Nfft) / Nfft) ** 2
freq = np.fft.fftfreq(sp.shape[-1])

sp = sp[:int(len(sp) / 2)]
freq = freq[:int(len(freq) / 2)]
freq = freq * SampPerRev

fig1 = plt.figure()
plt.plot(freq, sp)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

Bfq = 5.36
plt.axis([freq[0], 25, 0, 1.1 * max(sp[(freq >0) * (freq < 100)])])
xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
for xc in xcoords:
    plt.axvline(x=xc, color='red', linestyle=':')
plt.savefig('figures/Test_UN_PI_baseline00.pdf', bbox_inches='tight')
# %% loading data
filed = 'Data/Test_16_PI_STASK'
mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)

data_angle = mat_contents['data_angle_save']
datas = data_angle['angle_l1'][0,0][0]
SampPerRev = data_angle['SampPerRev_l1'][0,0][0]
#%% signal angle resampled - SK baseline 01
#% SK filtering Jerome
Nw       = 2**np.fix(np.log2(1/.4*SampPerRev))*2**3
Nw       = int(Nw)
Nfft     = 2*Nw
#Noverlap = round(0.945*Nw)
Noverlap = round(3/4*Nw)

SK,_,_,_ = SK_W(datas,Nfft,Noverlap,Nw)
b        = np.fft.fftshift(np.real(np.fft.ifft(SK)))
datas    = sg.fftconvolve(datas,b,mode='same')

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datas)**2
datae = datae-np.mean(datae)
sp = np.abs(np.fft.fft(datae,Nfft)/Nfft)**2
freq = np.fft.fftfreq(sp.shape[-1])

sp = sp[:int(len(sp)/2)]
freq = freq[:int(len(freq)/2)]
freq = freq*SampPerRev

fig1 = plt.figure()
plt.plot(freq, sp)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

Bfq = 5.36
plt.axis([freq[0], 25, 0, 1.1 * max(sp[(freq >0) * (freq < 100)])])
xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
for xc in xcoords:
    plt.axvline(x=xc, color='red', linestyle=':')
plt.savefig('figures/Test_UN_PI_baseline01.pdf', bbox_inches='tight')
# %% Proposed STASK
filed = 'Data/Test_16_PI_STASK'
mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)

SampPerRev = data_angle['SampPerRev_l1'][0,0][0]
freq = mat_contents['f_env'][0,:]*SampPerRev
sp = mat_contents['psd_env_save'][:,1]

fig1 = plt.figure()
plt.plot(freq, sp)
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

Bfq = 5.36
plt.axis([freq[0], 25, 0, 1.1 * max(sp[(freq >0) * (freq < 100)])])
xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
for xc in xcoords:
    plt.axvline(x=xc, color='red', linestyle=':')
plt.savefig('figures/Test_UN_PI_STASK.pdf', bbox_inches='tight')