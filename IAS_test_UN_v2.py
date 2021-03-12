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

"""db_names = ['Test_15_UNB',
 'Test_16_PI',
 'Test_17_PO',
 'Test_18_BALL',
 'Test_19_UNDvar',
 'Test_20_UND_const']"""
k = 3
savename = db_names[k] + '_STASK'
f_est_save = np.zeros([114688, 14])
data_save = np.zeros([114688, 14])
data_angle_save = {}
psd_env_save = np.zeros([2 ** 14, 14])
for l in np.r_[:14]:
    data_ori = mat_contents[db_names[k]][:, l]
    # decimate
    dr = 16  # decimate factor
    data = sg.decimate(data_ori, dr)
    fs = fs_ori / dr
    # % estimating IAS
    fmaxmin = np.r_[10, 40] / fs
    delta = 5 / fs
    Nw = 2 ** np.ceil(np.log2(1 / fmaxmin[0] * 2))  # window length
    Nw = int(Nw)
    K = fs / 2 / 40;
    K = int(K)
    K = np.arange(K, 1, -1)
    K = np.r_[4]
    f_alpha_est = iaslinearapproxv2(data, Nw, K, fmaxmin, delta, 0, 'tmpaccxv3')

    data_cut = data_ori[:int(Nw * dr + (f_alpha_est.shape[0] - 1) * (Nw * dr / 2))]
    data_cut = data_cut[int(Nw * dr / 2):-int(Nw * dr / 2)]
    t2 = np.r_[:len(f_alpha_est)] / fs_ori * (Nw * dr / 2)
    pol = interpolate.interp1d(t2, f_alpha_est[:, 0])
    # % to length original 8 times sample
    tf = np.r_[:len(data_cut)] / fs_ori
    f_est = fs * pol(tf)
    f_est_save[:, l] = f_est
    data_save[:, l] = data_cut
    # % COT and STSK
    SampPerRev = 2 ** np.fix(np.log2(fs_ori / 2 / max(f_est))) * 2 ** 2
    datas_cut, tc, SampPerRev, _ = COT_intp2(f_est, SampPerRev, data_cut, fs_ori)

    s_speed = 0.3913
    Nw1 = int(2 ** np.fix(np.log2(1 / s_speed * SampPerRev)) * 2 **3)

    Nw2: int = 2 ** 8
    Nfft2 = 2 * Nw2

    Nfft = 2 * Nw1
    Noverlap = round(3 / 4 * Nw1)
    Window = np.kaiser(Nw1, beta=0)  # beta 0 rectangular,5	Similar to a Hamming
    # 6	Similar to a Hanning, 8.6	Similar to a Blackman

    psd_env, f_env, K, _ = PSD_envW(datas_cut, Nfft, Noverlap, Window, Nw2, Nfft2, 1)

    data_angle_save['angle_l' + str(l)] = datas_cut
    data_angle_save['SampPerRev_l' + str(l)] = SampPerRev
    psd_env_save[:, l] = psd_env

result_dict = {'f_est_save': f_est_save, 'data_save': data_save, \
               'data_angle_save': data_angle_save, 'psd_env_save': psd_env_save, \
               'tf': tf, 'fs_ori': fs_ori, 'f_env': f_env}

sio.savemat('Data/' + savename + '.mat', result_dict, do_compression=True)
# k =1 r0,r13 no alla f_est
# k =2 todos ok f_est
# k =3 r0,r2,r5,r7,r9,r11