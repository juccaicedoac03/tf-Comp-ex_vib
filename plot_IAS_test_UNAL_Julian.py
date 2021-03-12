import os
import sys
import scipy.io as sio  # load save matlab files
import numpy as np  # np matrix functions
import scipy.signal as sg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt  # plotting
from matplotlib.gridspec import GridSpec  # easy subplots
import pickle
import scipy.spatial.distance as dis
plt.rc('font', size=14)  # default text size
#%%
#sys.path.append('drive/My Drive/Detección_de_fallas')
#sys.path.append('drive/My Drive/Detección_de_fallas/STSK')
#%%
data_names = ['Test_18_BALL_STASK', 'Test_16_PI_STASK', 'Test_17_PO_STASK']  # 'Test_17_PO_STASK'

cont = 0
for i in data_names:
    filed = 'Data/' + i + '.mat'
    mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
    ent = pickle.load(open("Results/" + i + ".p", "rb"))
    entropy = ent['entropy']

    if cont == 0:
        data_angle = mat_contents['data_angle_save']
        datas = data_angle['angle_l4'][0, 0][0]
        SampPerRev = data_angle['SampPerRev_l4'][0, 0][0]
        fs = int(mat_contents['fs_ori'])
        Nw = 2 ** np.fix(np.log2(1 / .4 * SampPerRev)) * 2 ** 3
    elif cont == 1:
        data_angle = mat_contents['data_angle_save']
        datas = data_angle['angle_l1'][0, 0][0]
        SampPerRev = data_angle['SampPerRev_l1'][0, 0][0]
        Nw = 2 ** np.fix(np.log2(1 / .4 * SampPerRev)) * 2 ** 3
    else:
        fs = int(mat_contents['fs_ori'])
        data = mat_contents['data_save'][:, 2]  ### data
        f_est = mat_contents['f_est_save'][:, 2]
        Nw = int(2 ** np.fix(np.log2(1 / 0.39 * fs)))

    Nw = int(Nw)
    Nfft = 2 * Nw
    Noverlap = round(3 / 4 * Nw)

    E = np.concatenate((entropy, entropy[::-1]), axis=0)
    b = np.fft.fftshift(np.real(np.fft.ifft(E)))
    datas = sg.fftconvolve(datas, b, mode='same')

    # envelope spectrum
    datae = sg.hilbert(datas)
    datae = np.abs(datas) ** 2
    datae = datae - np.mean(datae)
    sp = np.abs(np.fft.fft(datae, Nfft) / Nfft) ** 2
    freq = np.fft.fftfreq(sp.shape[-1])

    sp = sp[:int(len(sp) / 2)]
    freq = freq[:int(len(freq) / 2)]
    freq = freq * SampPerRev

    if cont == 0:
        fig1 = plt.figure()
        plt.plot(freq, sp)
        plt.xlabel('Order [xn]')
        plt.ylabel('PSD')
        print(1.1 * max(sp[(freq > 0) * (freq < 100)]))

        Bfq = 2.19
        plt.axis([freq[0], 5, 0, 1.1 * max(sp[(freq > 0) * (freq < 100)])])
        xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
        plt.vlines(x=xcoords, ymin=0, ymax=1e-2, color='red', linestyle=':', label='BSF')

        Bfq = .4
        plt.axis([freq[0], 5, 0, 1.1 * max(sp[(freq > 0) * (freq < 100)])])
        xcoords = np.r_[:30 * Bfq:Bfq] + Bfq
        plt.vlines(x=xcoords, ymin=0, ymax=1e-2, color='green', label='FTF', linestyle='dashdot')
        plt.legend(loc='upper right', shadow=True)

    elif cont == 1:
        fig1 = plt.figure()
        plt.plot(freq, sp)
        plt.xlabel('Order [xn]')
        plt.ylabel('PSD')

        Bfq = 5.36
        plt.axis([freq[0], 25, 0, 1.1 * max(sp[(freq > 0) * (freq < 100)])])
        xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
        for xc in xcoords:
            plt.axvline(x=xc, color='red', linestyle=':')
    else:
        fig1 = plt.figure()
        plt.plot(freq, sp)
        plt.xlabel('Order [xn]')
        plt.ylabel('PSD')

        Bfq = 3.4
        plt.axis([freq[0], 25, 0, 1.1 * max(sp[(freq > 0) * (freq < 100)])])
        xcoords = np.r_[:12 * Bfq:Bfq] + Bfq
        for xc in xcoords:
            plt.axvline(x=xc, color='red', linestyle=':')
    plt.savefig('figures/' + i[:-6] + '_entropy.pdf', bbox_inches='tight')
    cont += 1
