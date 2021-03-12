# %%

import os
import sys
import pickle
import scipy.spatial.distance as dis
from scipy.signal import stft
import numpy as np  # math pack!
import pandas as pd  # subplots?
import scipy.signal as sg  # DSP
from scipy import integrate  # integration pack
from numpy import linalg as LA  # Linear algebra
from scipy import interpolate  # interpolate lib
import sys  # sio pack
import matplotlib.pyplot as plt  # plotting
import time  # tic toc
import scipy.optimize as optm  # optimization pack
import scipy.io as sio  # load save matlab files


# %%

def noise_snr(sig, reqSNR, color):
    sigPower = np.sum(np.abs(sig ** 2)) / sig.size
    reqSNR = np.power(10, reqSNR / 10)
    noisePower = sigPower / reqSNR
    if color == 'pink':
        noise = np.sqrt(noisePower) * pink(sig.size)
    if color == 'white':
        noise = np.sqrt(noisePower) * np.random.randn(sig.size)
    sig = sig + noise
    return sig


def bearingfault(dsurD, N, fo, fi, typeF, fr, fe, T):
    # [out,TPT] = bearingfault(dsurD, N, fo,fi,type,fr,fe,T)
    # dsurD : rapport entre diametre de bille et primitif
    # N : nombre de billes
    # fo : fréquence de rotation bague externe (si vecteur 2*1, fréquence début et fin)
    # fi : fréquence de rotation bague interne (si vecteur 2*1, fréquence début et fin)
    # type : 0:'BPFO' pour défaut de bague externe
    #       1:'BPFI' pour défaut de bague interne
    # fr : fréquence de résonance structure
    # fe : fréquence d'échantillonnage
    # T : durée du signal généré
    TPT = 44;
    eps = 0.2;
    t = np.linspace(0, T, T * fe)

    if not isinstance(fo, int):
        Fo = fo[0] + t * (np.diff(fo) / T)
    else:
        Fo = np.ones(t.shape) * fo

    if not isinstance(fi, int):
        Fi = fi[0] + t * (np.diff(fi) / T)
    else:
        Fi = np.ones(t.shape) * fi

    Fc = 1 / 2 * ((1 - dsurD) * Fi + (1 + dsurD) * Fo)

    tetao = np.cumsum(Fo) / fe
    tetai = np.cumsum(Fi) / fe
    tetac = np.cumsum(Fc) / fe

    out = np.zeros([2 * len(t), 2])
    out[:len(t), 0] = np.sin(2 * np.pi * TPT * (tetao - tetai))

    timpend = np.log(100) / (eps * 2 * np.pi * fr)
    timp = np.linspace(0, timpend, int(np.ceil(timpend * fe)))

    impulse = np.sin(2 * np.pi * fr * timp) * np.exp(-eps * 2 * np.pi * fr * timp)
    # impulse = zeros(size(timp));
    # impulse(1)=1;

    nimpulse = len(impulse)
    BPFO = N / 2 * (1 - dsurD)
    BPFI = N / 2 * (1 + dsurD)

    print('BPFO=', np.abs(BPFO))
    print('BPFI=', np.abs(BPFI))

    idef, modulation = faill_type(typeF, N, tetac, tetao, tetai)
    # idef = round(tdef*fe)+1;

    for ii in np.r_[:len(idef)]:
        out[idef[ii] + np.r_[:nimpulse], 1] = impulse.reshape(-1, 1)[:, 0]  # + randn(nimpulse,1)

    out = out[:len(t), :]

    out[:, 1] = out[:, 1] * modulation

    # bruit
    # out(:,2) = out(:,2) + randn(size(out(:,2))).*std(out(:,2));
    return out, TPT


# % auxiliar functions switch bearing failure model
def BPFO(N, tetac, tetao, tetai):
    idef = np.flatnonzero(np.abs(np.diff(np.round(N * (tetac - tetao)))))
    modulation = np.cos(2 * np.pi * tetao.reshape(-1, 1)[:, 0])
    return idef, modulation


def BPFI(N, tetac, tetao, tetai):
    idef = np.flatnonzero(np.abs(np.diff(np.round(N * (tetac - tetai)))))
    modulation = np.cos(2 * np.pi * tetai.reshape(-1, 1)[:, 0])
    return idef, modulation


def faill_type(argument, N, tetac, tetao, tetai):
    switcher = {
        0: BPFO,
        1: BPFI
    }
    # Get the function from switcher dictionary
    func = switcher.get(argument, "Press 0 or 1")
    # Execute the function
    return func(N, tetac, tetao, tetai)


# %%
def tTacho_fsig(rpm, fs, PPR, isencod):
    # rpm tacho signal
    # fs sampling frequency
    # PPR pulses per revolition, resolution tacho 44 surveillance case
    # isencod ==1 encoder signal or IF profile
    # sm smoothing intervals with 20
    if isencod == 1:
        x = rpm
    else:
        x = np.sin(2 * np.pi * PPR * integrate.cumtrapz(rpm) / fs)
    t = np.arange(0, len(x))
    t = t / fs
    # Produce +1 where signal is above trigger level
    # and -1 where signal is below trigger level
    TLevel = 0
    xs = np.sign(x - TLevel)
    # Differentiate this to find where xs changes
    # between -1 and +1 and vice versa
    xDiff = np.diff(xs);
    # We need to synchronize xDiff with variable t from the
    # code above, since DIFF shifts one step
    tDiff = t[1:]
    # Now find the time instances of positive slope positions
    # (-2 if negative slope is used)
    tTacho = tDiff[xDiff.T == -2]  # xDiff.T return the indexes boolean
    # Count the time between the tacho signals and compute
    # the RPM at these instances
    rpmt = 60 / PPR / np.diff(tTacho);  # Temporary rpm values
    # Use three tacho pulses at the time and assign mean
    # value to the center tacho pulse
    rpmt = 0.5 * (rpmt[0:-1] + rpmt[1:]);
    tTacho = tTacho[1:-1];  # diff again shifts one sample

    wfiltsv = int(2 ** np.fix(np.log2(.05 * fs))) - 1
    rpmt = sg.savgol_filter(rpmt, wfiltsv, 2)  # smoothing filter

    rpmt = interpolate.InterpolatedUnivariateSpline \
        (tTacho, rpmt, w=None, bbox=[None, None], k=1)

    rpmt = rpmt(t)
    return rpmt, tTacho, xs


# % Angular resampling given tTacho
def COT_intp(x, tTacho, PPR, fs, SampPerRev):
    tTacho = tTacho[::PPR]  # Pick out every PPR pulse
    ts = np.r_[:0]  # Synchronous time instances

    for n in np.r_[:len(tTacho) - 1]:
        tt = np.linspace(tTacho[n], tTacho[n + 1], int(SampPerRev) + 1)
        ts = np.r_[ts, tt[:-1]]
    # Now upsample the original signal 10 times (to a total
    # of approx 25 times oversampling).
    x = sg.resample(x, 10 * len(x))
    fs = 10 * fs
    # create a time axis for this upsampled signal
    tx = np.r_[:len(x)] / fs
    # Interpolate x onto the x-axis in ts instead of tx
    xs = interpolate.InterpolatedUnivariateSpline(tx, x, w=None, bbox=[None, None], k=1)
    xs = xs(ts)
    tc = np.r_[:len(xs) / SampPerRev:1 / SampPerRev]
    return xs, tc, SampPerRev


# %%

# type : 0:'BPFO' pour dÃ©faut de bague externe
#       1:'BPFI' pour dÃ©faut de bague interne
dsurD = 15 / 32;
N = 8;
fo = 0;
fi = np.r_[5, 30] * 5; #error
typeF = 1;
fr = 5e3;
fs = 2 ** 14;
T = 10;
data, TPT = bearingfault(dsurD, N, fo, fi, typeF, fr, fs, T)
# %%
f0e = data[:, 0]
f0 = sg.hilbert(f0e)
f0 = np.diff(np.unwrap(np.angle(f0))) * fs / TPT / (2 * np.pi)
f0 = np.r_[f0, f0[-1]]
wfiltsv = int(2 ** np.log2(int(.05 * fs)) - 1)
f0 = sg.savgol_filter(f0, wfiltsv, 2)  # smoothing filter
data = data[:, 1]
data = noise_snr(data, 3, 'white')
t = np.r_[:len(data)] / fs

# %% Angular resampling & envelope spectrum Baseline 01
isencod = 1
_, tTacho, _ = tTacho_fsig(f0e, fs, TPT, isencod)
SampPerRev = 2 ** np.fix(np.log2(fs / 2 / fi[1])) * 2 * 2
datas, tc, SampPerRev = COT_intp(data, tTacho, TPT, fs, SampPerRev)

fs = SampPerRev
# SK filtering
Nw = 2 ** np.fix(np.log2(SampPerRev * 1 / 2)) * 2 ** 3  # 1/2 minimun expected frq 2Hz
Nw = int(Nw)
Nfft = 2 * Nw
# Noverlap = round(0.945*Nw)
Noverlap = round(3 / 4 * Nw)
# %%

ent = pickle.load(open("Results/numerical.p", "rb"))
entropy = ent['entropy']
# %%

E = np.concatenate((entropy, entropy[::-1]), axis=0)
b = np.fft.fftshift(np.real(np.fft.ifft(E)))
datas = sg.fftconvolve(datas, b, mode='same')
# %%

# envelope spectrum
datae = sg.hilbert(datas)
datae = np.abs(datae) ** 2
datae = datae - np.mean(datae)
# %%

# % plot envelope spectrum
fig = plt.figure()
sp = np.abs(np.fft.fft(datae) / len(datae)) ** 2
sp = sp[:int(len(sp) / 2)]
freq = np.fft.fftfreq(datae.shape[-1])
freq = freq[:int(len(freq) / 2)]
freq = freq * SampPerRev
plt.plot(freq, sp)
plt.axis([freq[0], SampPerRev / 2, 0, 1.1 * max(sp[(freq > 2) * (freq < 100)])])
plt.xlabel('Order [xn]')
plt.ylabel('PSD')

Bfq = 5.875
xcoords = np.r_[:16 * Bfq:Bfq] + Bfq
plt.vlines(x=xcoords, ymin=0, ymax=1.5 * max(sp), color='red', label='BPFI', linestyle='dashdot')
plt.legend(loc='upper right', shadow=True)
plt.savefig('Figures/numerical_entropy_.pdf', bbox_inches='tight')
