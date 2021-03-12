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
from functions.STSK import COT_intp2, PSD_envW

# %% loading data
filed = 'Data/DB_test_UN'
mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
fs_ori = 25.6e3

db_names = list(mat_contents.keys())
db_names = [l for l in db_names if '__' not in l]
data_ori = mat_contents[db_names[3]][:, 4]
#del mat_contents

# decimate
dr = 16  # decimate factor
data = sg.decimate(data_ori, dr)
fs = fs_ori / dr
t = np.r_[:len(data)] / fs
# %% spectrogram
plt.close('all')
NFFT = 2 ** 9  # the length of the windowing segments

fig = plt.figure(figsize=(10, 6))  # constrained_layout=true
gs = GridSpec(5, 5, wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[:1, :])
ax2 = fig.add_subplot(gs[1:, :])

ax1.plot(t, data)
ax1.set_xlim([t[0], t[-1]])
ax1.tick_params(
    axis='both',  # changes apply to the both-axis
    which='both',  # both major and minor ticks are affected
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off

# spectrogram
Pxx, freqs, bins, im = ax2.specgram(data, NFFT=NFFT, Fs=fs, noverlap=NFFT * 0.95, cmap='viridis')
ax2.set_ylim([0, fs / 2])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Frequency [Hz]')

plt.draw()
plt.savefig('figures/test_UN_spectrogram_original.pdf', bbox_inches='tight')
# %% estimating IAS

fmaxmin = np.r_[10, 40] / fs
delta = 5 / fs

Nw = 2 ** np.ceil(np.log2(1 / fmaxmin[0] * 2))  # window length
Nw = int(Nw)

K = fs / 2 / 40; K = int(K)
K = np.arange(K, 1, -1)
K = np.r_[4]
f_alpha_est = iaslinearapproxv2(data, Nw, K, fmaxmin, delta, 0, 'tmpaccxv3')
#plt.plot(f_alpha_est[:,32])
#%% f_est interpolation and cutting original data
#f_est = sg.resample(f_alpha_est, data[:-Nw].shape[0])
data_cut = data_ori[:int(Nw*dr+(f_alpha_est.shape[0]-1)*(Nw*dr/2))]
data_cut = data_cut[int(Nw*dr/2):-int(Nw*dr/2)]
t2 = np.r_[:len(f_alpha_est)]/fs_ori*(Nw*dr/2)
pol = interpolate.interp1d(t2, f_alpha_est[:, 0])
#%% to length original 8 times sample
tf = np.r_[:len(data_cut)]/fs_ori
f_est = fs*pol(tf)

plt.figure()
plt.plot(tf, f_est)
plt.xlabel('Time[s]')
plt.ylabel('Frequency [Hz]')
#%% COT and STSK
SampPerRev = 2 ** np.fix(np.log2(fs_ori / 2 / max(f_est[int(fs_ori / 2):-int(fs_ori / 2)]))) * 2 ** 2
datas_cut, tc, SampPerRev, _ = COT_intp2(f_est, SampPerRev, data_cut, fs_ori)

s_speed = 0.3913
Nw1 = int(2 ** np.fix(np.log2(1 / s_speed * SampPerRev)) * 2 ** 3)

Nw2: int = 2 ** 8
Nfft2 = 2 * Nw2

Nfft = 2 * Nw1
Noverlap = round(3 / 4 * Nw1)
Window = np.kaiser(Nw1, beta=0)  # beta 0 rectangular,5	Similar to a Hamming
# 6	Similar to a Hanning, 8.6	Similar to a Blackman

psd_env, f, K, _ = PSD_envW(datas_cut, Nfft, Noverlap, Window, Nw2, Nfft2, 1)

#%% plot
plt.figure()
plt.plot(f*SampPerRev, psd_env)
plt.xlim([0, 20])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

# plot psd signal in angular domain
plt.figure()
nfft = 2**12
psd = np.fft.fft(sg.hilbert(datas_cut), nfft)/nfft
psd = np.abs(psd)**2
freq = np.fft.fftfreq(psd.shape[-1])
psd = psd[:int(len(psd)/2)]
freq = freq[:int(len(freq)/2)]
plt.plot(freq*SampPerRev, psd)
plt.xlim([0, 100])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

plt.figure()
plt.plot(tc, datas_cut)
plt.xlim([tc[0], tc[-1]])
plt.title('Signal in angular domain')
plt.xlabel('angle [rad]')

plt.figure()
plt.plot(tf, data_cut)
plt.xlim([tf[0], tf[-1]])
plt.title('Signal in time domain')
plt.xlabel('Time [s]')
#%%_______________
# #%% f_est approximation
# n = 0
# f_est = f_alpha_est[:, 2 * n] * fs
# alpha_est = f_alpha_est[:, 2 * n + 1] * fs
# f_est_lin = f_est_linear(alpha_est, f_est, Nw, H=Nw/2)
# t = np.r_[:len(f_est_lin)] / fs
