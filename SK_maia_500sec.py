import os  # change dir
import numpy as np
import scipy.io as sio  # load save matlab files
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mcolors

from functions.STSK import PSD_envW


# function load matfile
def loading_matfile(filed='none', flat=True):
    mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
    for k, v in mat_contents.items():
        if hasattr(v, 'flatten') and flat is True:
            globals()[k] = v.flatten()
        else:
            globals()[k] = v
    return globals()


def load_mat_dic(file=str) -> dict:
    matdict = sio.loadmat(file, mdict=None, appendmat=True)
    return {k: v.flatten() for k, v in matdict.items() if k[0] != '_'}


plt.rc('font', size=14)  # default text size
# plt.rcParams["font.family"] = "Times New Roman" # font
# %% loading signal angle resampled - SK filtering proposal
# loading_matfile(filed='Data/SK-maia_500_angle_domain', flat=True)
# # loading_matfile(filed='Data/SK-maia_angle_domain')
#
# all_values_dic = sio.loadmat('Data/maia_500_angle_domain_speeds', mdict=None, appendmat=True)
# shaft_str = ['shaft 7', 'shaft 5', 'shaft 3 (sol)', 'carrier', 'planetary gear speed', 'BPF']
#
# datas = datas - np.mean(datas)
# K = 0
#
# # % SK filtering proposal
# NW1 = [
#     int(2 ** np.fix(np.log2(1 / s_speed * SampPerRev)) * 2 ** 4)
#     for key, s_speed in all_values_dic.items()
#     if key in shaft_str
# ]
# Nw2: int = 2 ** 8
# Nfft2 = 2 * Nw2
#
# psd_all = dict()
# f_all = dict()
# for name, Nw1 in zip(shaft_str, NW1):
#     Nfft = 2 * Nw1
#     Noverlap = round(3 / 4 * Nw1)
#     Window = np.kaiser(Nw1, beta=0)  # beta 0 rectangular,5	Similar to a Hamming
#     # 6	Similar to a Hanning, 8.6	Similar to a Blackman
#
#     psd, f, K, _ = PSD_envW(datas, Nfft, Noverlap, Window, Nw2, Nfft2, 1)
#     psd_all.update({name: psd})
#
# psd_all.update({'SampPerRev': SampPerRev})
# sio.savemat('Data/maia_500_SK_PSDs.mat',
#             psd_all, do_compression=True)
# %% plotting envelope analysis
all_values_dic = load_mat_dic('Data/maia_500_angle_domain_speeds') # dictionary with speed values
shaft_str = ['shaft 7', 'shaft 5', 'shaft 3 (sol)', 'carrier', 'planetary gear speed', 'BPF'] # speed values to look in dict

psd_all = load_mat_dic('Data/maia_500_SK_PSDs') # all SK psd computed
SampPerRev = psd_all['SampPerRev'].item(0) # always do item(0) for constants
psd_all.pop('SampPerRev')

plt.close('all')
linestyle_str = [
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'
linestyle_str = [lstr for name, lstr in linestyle_str]
linestyle_str = linestyle_str[::-1]
color_s = [color for color in mcolors.BASE_COLORS.keys() if color not in ['w', 'r']]
color_s.append('r')
color_s = color_s[::-1]

BPFI = [shaft_speed * 9.306 for key, shaft_speed in all_values_dic.items() if key in shaft_str]
BPFO = [shaft_speed * 6.694 for key, shaft_speed in all_values_dic.items() if key in shaft_str]
BSF = [shaft_speed * 2.981 for key, shaft_speed in all_values_dic.items() if key in shaft_str]
B_fault = {'BPFI': BPFI, 'BPFO': BPFO, 'BSF': BSF}
del BPFI, BPFO, BSF

for K, key in enumerate(psd_all):
    f = np.linspace(0, SampPerRev / 2, psd_all[key].size,endpoint=True)
    plt.figure()
    plt.plot(f, psd_all[key])
    plt.xlabel('Order [xn]')
    plt.ylabel('PSD')
    xlims = [B_fault[name][K] for name in B_fault]
    xlims = [0.9 * min(xlims), max(xlims) * 6.25]
    psd_lim = max(psd_all[key][(f > xlims[0]) * (f < xlims[1])])
    plt.axis([0, xlims[1], 0, 1.2 * psd_lim])

    for k, (f_name, f_freq) in enumerate(B_fault.items()):
        f_freq = f_freq[K]
        kpoint = int(f[-1] / f_freq) + 1
        xcoords = f_freq * np.r_[1:kpoint + 1]
        plt.vlines(x=xcoords, colors=color_s[k], linestyle=linestyle_str[k], linewidth=1.5, ymin=0, \
                   ymax=1.2 * max(psd_all[key]), label=f_name + ' x ' + key)
    plt.legend(loc='upper right', shadow=True)
    save_dir = f'figures/maia_500_env_analysis_{key}.pdf'  # f-string
    save_dir = save_dir.replace(" ", "")
    plt.savefig(save_dir, bbox_inches='tight')
    plt.draw()
plt.show()
# # tips pychar
# # ctrl+p parameters in the parentheses for a funciton
# # ctrl+may+f7 highlight variable in all document, f3 and f3 to navigate
# # ctrl+espace type any characters
# # code completion commands ctrl+Q, ctrl+P, ctrl+B
