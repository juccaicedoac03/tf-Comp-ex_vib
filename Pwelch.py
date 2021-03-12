from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from functions.COT import tTacho_fsig,COT_intp

fs = 10e3
N = 1e5
amp = 2*np.sqrt(2)
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = signal.welch(x, fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
#%% only noise
fs = 10e3
N = 1e5
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
#x = amp*np.sin(2*np.pi*freq*time)
x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
#COT_intp(x,tTacho,PPR,fs,SampPerRev)
#x = x+np.exp(1j*2*np.pi*2e3*time)
dtx = np.cos(2*np.pi*(200*time/2+100)*time)#deterministic
#x = x+dtx
x = x*dtx
f, Pxx_den = signal.welch(x, fs, nperseg=512)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlim([0, fs/2])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
#%%
f, t, Zxx = signal.stft(x, fs=fs, nperseg=2**8)
#% ploting STFT
plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx))
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()