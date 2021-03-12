# importing libraries
import matlab.engine
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt  # plotting

# plt.rc('font', size=14)  # default text size
plt.rcParams["font.family"] = "Times New Roman"  # font

# paths for matlab user functions
DRS_toolbox = r'G:\Mon Drive\GCPDS\Toolbox\Jerome Toolboxes\DRS'
# %% Example starts here

# Synthesise signal
N = 2 ** 16
t = np.r_[0:N]
p = np.sin(2 * np.pi * t / 17.3) + .5 * np.sin(
    2 * np.pi * t / 11)  # periodic part (two perdiods of 17 and 11 samples each)
n = .5 * np.random.randn(N)  # random part (here white noise)
x = p + n

# Set the critical parameters
delay = 10  # any delay > 0 should work here since the correlation length of white noise is 0
Nwind = 512  # should be grater than 10~20 times the longest period, i.e. 170~340
# %% Estimate the separation filter (frequency gain) G Matlab code
eng = matlab.engine.start_matlab()  # starting engine
eng.addpath(DRS_toolbox, nargout=0)  # adding user functions to matlab path

if isinstance(x, np.ndarray):
    x = matlab.double(x.tolist())

G = eng.DRS(x, delay, Nwind)  # always transform numpy data to list

# Estimate the periodic part from filtering with G
p_est = eng.Filt_STFT(x, eng.transpose(G))

# Estimate the periodic part from filtering with G
p_est2 = eng.Filt_STFT(x, eng.transpose(eng.abs(G)))

# returning matlab variables to python variable
p_est = np.asarray(p_est)
p_est = p_est.flatten()
p_est2 = np.asarray(p_est2)
p_est2 = p_est2.flatten()
x = np.asarray(x)
x = x.flatten()
G = np.asarray(G)
G = G.flatten()

# Estimate the noise part by substraction using complex DRSep Gain
N2 = Nwind + delay  # (accounts for filter delay!!)
n_est = x[N2:N] - p_est[:N - N2]

# Estimate the noise part by substraction using magnitude of DRSep Gain
n_est2 = x - p_est2

# Gain to numpy array

eng.quit()  # stop engine
# %% Plotting
# plotting Gain
f = np.r_[:Nwind] / Nwind
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(211)
ax1.plot(f, np.abs(G[:Nwind]), 'k')
ax1.grid(True)
ax1.title.set_text('magnitude of DRSep Gain')
ax2 = fig.add_subplot(212)
ax2.plot(f, np.angle(G[:Nwind]), 'k')
ax2.grid(True)
ax2.title.set_text('phase of DRSep Gain')

# plotting reconstructed signals in time
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(t, x, 'k')
plt.grid(True)
plt.title('original signal')
plt.subplot(312)
plt.plot(t, p, 'k')
plt.plot(t[N2:N], p_est[:N - N2], 'r')
plt.title('original periodic part + estimated one')
plt.subplot(313)
plt.plot(t, n, 'k')
plt.plot(t[N2:N], n_est[:N - N2])
plt.title('original random part + estimated one')

# plotting reconstructed signals in frequency
psd_args = {'color': 'k', 'NFFT': 2 ** 10}
plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.psd(x, **psd_args)
plt.title('PSD original signal')
plt.subplot(312)
plt.psd(n_est[:N - N2], **psd_args)
psd_args['color'] = 'r'
plt.psd(p_est[:N - N2], **psd_args)
plt.title('PSD estimated periodic and noise parts (from complex DRSep Gain)')
plt.subplot(313)
psd_args['color'] = 'k'
plt.psd(n_est2[:N - N2], **psd_args)
psd_args['color'] = 'r'
plt.psd(p_est2[:N - N2], **psd_args)
plt.title('PSD estimated periodic and noise parts (from magnitude of DRSep gain)')
