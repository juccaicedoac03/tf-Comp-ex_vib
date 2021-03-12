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
plt.rc('font', size=14) # default text size
#%% load data
filed = 'Data/Test_16_PI_STASK'
mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
data_angle = mat_contents['data_angle_save']

# %% spectrogram angle

plt.close('all')

data_ori = data_angle['angle_l1'][0,0][0]
fs_ori = data_angle['SampPerRev_l1'][0,0][0]
# decimate
dr = 32  # decimate factor
data = sg.decimate(data_ori, dr)
fs = fs_ori / dr
t = np.r_[:len(data)]/fs
NFFT = 2 ** 8  # the length of the windowing segments

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
ax2.set_ylim([0, fs/2])
ax2.set_xlabel('Time [sec]')
ax2.set_xlabel('Rotations [xn]')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_ylabel('Order [xn]')

plt.draw()
plt.savefig('figures/Test_UN_PI_spec_angle.pdf',bbox_inches='tight')
# %% spectrogram time

plt.close('all')

data_ori= mat_contents['data_save'][:,1]
fs_ori = int(mat_contents['fs_ori'])
t = np.r_[:len(data)]/fs

# decimate
dr = 32  # decimate factor
data = sg.decimate(data_ori, dr)
fs = fs_ori / dr
t = np.r_[:len(data)] / fs

NFFT = 2 ** 8  # the length of the windowing segments

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
ax2.set_ylim([0, fs/2])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Frequency [Hz]')

plt.draw()
plt.savefig('figures/Test_UN_PI_spec.pdf',bbox_inches='tight')
#%% plot f_est
f_est = mat_contents['f_est_save'][:,1]
t = np.r_[:len(f_est)]/fs_ori
plt.plot(t,f_est)
plt.xlabel('Time [sec]')
plt.ylabel('Frequency [Hz]')
plt.xlim([t[0],t[-1]])
plt.savefig('figures/Test_UN_PI_f_est.pdf',bbox_inches='tight')
#%% plot psds
# plot psd signal in angular domain
data_ori= mat_contents['data_save'][:,1]

plt.figure()
plt.subplot(211)
nfft = 2**12
psd = np.fft.fft(sg.hilbert(data_ori), nfft)/nfft
psd = np.abs(psd)**2
freq = np.fft.fftfreq(psd.shape[-1])
psd = psd[:int(len(psd)/2)]
freq = freq[:int(len(freq)/2)]
plt.plot(freq*fs_ori, psd)
plt.xlim([0, fs_ori/2])
plt.ylim([0, 1e-2])
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD')

plt.subplot(212)
data_ori = data_angle['angle_l1'][0,0][0]
fs_ori_ang = data_angle['SampPerRev_l1'][0,0][0]
nfft = 2**12
psd = np.fft.fft(sg.hilbert(data_ori), nfft)/nfft
psd = np.abs(psd)**2
freq = np.fft.fftfreq(psd.shape[-1])
psd = psd[:int(len(psd)/2)]
freq = freq[:int(len(freq)/2)]
plt.plot(freq*fs_ori_ang, psd)
plt.xlim([0, fs_ori_ang/2])
plt.ylim([0, 1e-2])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')
plt.savefig('figures/Test_UN_PI_psds.pdf',bbox_inches='tight')