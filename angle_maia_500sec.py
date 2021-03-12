import os  # change dir
import sys

Wdir = os.getcwd()
sys.path.append(Wdir + '\functions_stsk')  # add to path
dbdir = 'G:/Mon Drive/GCPDS Database/Case_western_N1N2'
import numpy as np
import scipy.io as sio  # load save matlab files
import scipy.signal as sg  # signal pack
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  # plotting
import matplotlib.colors as mcolors

from matplotlib.gridspec import GridSpec  # easy subplots
from scipy import interpolate  # interpolate lib

from functions.STSK import tTacho_fsig, COT_intp, SK_W, PSD_envW, \
    COT_intp2

plt.rc('font', size=14)  # default text size


# function load matfile
def loading_matfile(filed='none', flat=True):
    mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
    for k, v in mat_contents.items():
        if hasattr(v, 'flatten') and flat is True:
            globals()[k] = v.flatten()
        else:
            globals()[k] = v
    return globals()


# plt.rcParams["font.family"] = "Times New Roman" # font
plt.close('all')
# %% loading data
dbdir = 'G:/Mon Drive/GCPDS/2019/Python Scripts/STSK/Data'
mat_contents = sio.loadmat(dbdir + '/' + 'Mopa_maia', mdict=None, appendmat=True)
datab = {}
del mat_contents['__globals__']
del mat_contents['__header__']
del mat_contents['__version__']
for k, v in mat_contents.items():  # keys,values
    datab[k.lower()] = v
del mat_contents, k, v
f0 = datab['fve'].T
t_f0 = datab['t'].T
fs_f0 = 1 / (t_f0[1] - t_f0[0])
fs = int(datab['fe'])
del datab

dbdir = 'G:/Mon Drive/GCPDS/Data Base/Wind Turbine CMMNO2014'
mat_contents = sio.loadmat(dbdir + '/' + 'v5k8bits', mdict=None, appendmat=True)
datab = {}
del mat_contents['__globals__']
del mat_contents['__header__']
del mat_contents['__version__']
for k, v in mat_contents.items():  # keys,values
    datab[k.lower()] = v
del mat_contents, k, v
data = datab['data'][:, 0]
t = np.r_[0:len(data)] / fs
del datab
t_if = np.array([np.where(t == t_f0[0])[0][0], np.where(t == t_f0[-1])[0][0]])
t_if = t_if.astype(int)
t_f0 = np.r_[0:len(f0)] / fs_f0
data = data[t_if[0]:t_if[1]]
t = np.r_[0:len(data)] / fs

# interpolating IF profile obtained by MOPA
s_f0 = interpolate.InterpolatedUnivariateSpline(t_f0, f0, w=None, bbox=[None, None], k=3)
f0 = s_f0(t)
# f0 = sg.savgol_filter(f0, 2**14-1, 2)
# %% plotting spectrogram
plt.close('all')
NFFT = 2 ** 12  # the length of the windowing segments

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
Pxx, freqs, bins, im = ax2.specgram(data, NFFT=NFFT, Fs=fs, noverlap=NFFT / 2, cmap='viridis')
ax2.plot(t, 29 * f0, 'r')
ax2.set_xlim([t[0], t[-1]])
ax2.set_ylim([0, fs / 2])
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Frequency [Hz]')

plt.draw()
ax2.legend(['$f_{29}[n]$'])
plt.savefig('Figures\maia_500_spectrogram_original.pdf', bbox_inches='tight')
# %% angular resampling

SampPerRev = 2 ** np.fix(np.log2(fs / 2 / max(f0[int(fs / 2):-int(fs / 2)]))) * 2 ** 2

datas, tc, SampPerRev, _ = COT_intp2(f0, SampPerRev, data, fs)

# % Save dataz
result_dict = {'datas': datas, 'tc': tc, 'SampPerRev': SampPerRev, 'fs': fs}
sio.savemat('Data/SK-maia_500_angle_domain.mat',
            result_dict, do_compression=True)

# %% plotting spectrogram angular domain
loading_matfile(filed='Data/SK-maia_500_angle_domain', flat=True)
datas = sg.decimate(datas, 2)
SampPerRev = SampPerRev / 2
tc = np.r_[:len(datas)] / SampPerRev

# computing fft
datash = sg.hilbert(datas)
sp = np.fft.fft(datash) / len(datash)
sp = sp[:int(len(sp) / 2)]
sp = np.abs(sp) ** 2
freq = np.fft.fftfreq(tc.shape[-1])
freq = freq[:len(sp)] * SampPerRev

NFFT = int(2 ** np.fix(np.log2(1 / 0.025 * SampPerRev)))

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(5, 5, wspace=0.0, hspace=0.0)
ax1 = fig.add_subplot(gs[:1, 1:])
ax2 = fig.add_subplot(gs[1:, 1:])
ax3 = fig.add_subplot(gs[1:, :1])

ax1.plot(tc, datas)
ax1.set_xlim([tc[0], tc[-1]])
ax1.tick_params(
    axis='both',  # changes apply to the both-axis
    which='both',  # both major and minor ticks are affected
    labelbottom=False,
    labelleft=False)  # labels along the bottom edge are off

# spectrogram
# freqs, bins, Pxx = sg.spectrogram(datas, nperseg=NFFT,fs=SampPerRev, noverlap=NFFT/2)
# ax2.pcolormesh(bins*2*np.pi, freqs, 10*np.log10(Pxx),cmap='viridis')
Pxx, freqs, bins, im = ax2.specgram(datas, NFFT=NFFT, Fs=SampPerRev, noverlap=NFFT / 2, cmap='viridis')
ax2.set_xlim([tc[0], tc[-1]])
ax2.set_ylim([0, SampPerRev / 2])
ax2.set_xlabel('Rotations [xn]')
ax2.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    labelleft=False)  # labels along the bottom edge are off

ax3.plot(sp, freq)
ax3.set_ylim([0, SampPerRev / 2])
ax3.set_xlim([0, 1.2 * max(sp)])
ax3.set_ylabel('Order [xn]')
ax3.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    labelbottom=False)  # labels along the bottom edge are off
ax3.invert_xaxis()

plt.draw()
plt.savefig('Figures\maia_500_spectrogram_angular_specgram.pdf', bbox_inches='tight')
# %% ploting fig quentin I
# % loading signal angle resampled - SK filtering proposal
loading_matfile(filed='Data/SK-maia_500_angle_domain', flat=True)
# loading_matfile(filed='Data/SK-maia_angle_domain')
datas = datas.flatten()
datas = datas - np.mean(datas)
# computing fft
datash = sg.hilbert(datas)
sp = np.fft.fft(datash) / len(datash)
sp = sp[:int(len(sp) / 2)]
sp = np.abs(sp) ** 2
freq = np.fft.fftfreq(tc.shape[-1])
freq = freq[:len(sp)] * SampPerRev
freq = freq.flatten()
# %% ploting fig quentin II
# formula en https://www.wikimeca.org/index.php/Les_trains_%C3%A9picyclo%C3%AFdaux#Etude_cin.C3.A9matique_-_Formule_de_Willis
shaft_speeds_labels = ['shaft 7', 'shaft 5', 'shaft 3 (sol)', 'carrier', 'planetary gear speed']
N_t = {'N1': 123, 'N2': 50, 'N3': 21, 'N4': 93, 'N5': 22, 'N6': 120, 'N7': 29, 'N8': 63, 'N9': 23, 'N10': 10,
       'N11': 13}  # number of theets per gear
shaft_speeds = np.zeros(len(shaft_speeds_labels))
shaft_speeds[0] = 1  # shaft 7 wAR/0
shaft_speeds[1] = shaft_speeds[0] * (N_t['N7'] / N_t['N6'])  # shaft 5 wAI/0
shaft_speeds[2] = shaft_speeds[1] * (N_t['N5'] / N_t['N4'])  # shaft 3 wSOL/0
shaft_speeds[3] = shaft_speeds[2] / (1 + N_t['N1'] / N_t['N3'])  # Porte satelites Input shaft wPS/0
shaft_speeds[4] = shaft_speeds[2] * N_t['N3'] / (N_t['N1'] + N_t['N3']) * N_t['N1'] / N_t[
    'N2']  # planetary gear speed [w_sat/PS]
shaft_speeds_dic = {key: value for key, value in zip(shaft_speeds_labels, shaft_speeds)}

GMF_label = ['GMF 4-5', 'GMF 6-7', 'GMF 1-2', 'GMF 8-9', 'GMF 5-6', 'BPF']
GMF = np.zeros(len(GMF_label))
GMF[0] = N_t['N5'] * shaft_speeds_dic['shaft 5']  # 4-5 GMF = Z_P×RPM_P=Z_G×RPM_G
GMF[1] = N_t['N7'] * shaft_speeds_dic['shaft 7']  # GMF 6-7
# GMF[2] = N_t['N3'] * shaft_speeds_dic['shaft 3 (sol)']  # GMF 1-2 o 2-3
GMF[2] = N_t['N2'] * shaft_speeds_dic['planetary gear speed']  # GMF 1-2 o 2-3
GMF[3] = N_t['N8'] * shaft_speeds_dic['shaft 5']  # GMF 8-9
GMF[4] = N_t['N6'] * shaft_speeds_dic['shaft 5']  # GMF 5-6
GMF[5] = shaft_speeds_dic['carrier'] * 3  # BPF input shaft number of blades
GMF_label_dic = {key: value for key, value in zip(GMF_label, GMF)}

B_fault_label = ['BPFI', 'BPFO', 'BSF']
B_fault = np.zeros(len(B_fault_label))
B_fault[0] = 9.306
B_fault[1] = 6.694
B_fault[2] = 2.981
B_fault_dic = {key: value for key, value in zip(B_fault_label, B_fault)}

all_values_dic = {**shaft_speeds_dic, **GMF_label_dic, **B_fault_dic}
all_values_dic.update({'BPFIxPGS': all_values_dic['BPFI']*all_values_dic['planetary gear speed']})

linestyle_str = ['dashdot', 'dashed', 'dotted', 'dashdot', 'dashed', 'dotted']
color_s = {'GMF 4-5': 'r', 'GMF 6-7': 'b', 'GMF 1-2': 'g', 'GMF 8-9': 'm', 'shaft 5': 'b', 'BPF': 'r', 'carrier': 'c'}

figures_labels = [['GMF 4-5', 'GMF 6-7', 'GMF 1-2', 'GMF 8-9'], ['GMF 1-2', 'shaft 5', 'BPF'],
                  ['shaft 5', 'BPF', 'BPFI x carrier'], ['BPF', 'carrier'],

                  ['shaft 5', 'BPF', 'BPFI x BPF'], ['shaft 5', 'BPF', 'BPFI x carrier'],
                  ['shaft 5', 'BPF', 'BPFI x shaft 3 (sol)'],
                  ['shaft 5', 'BPF', 'BPFI x planetary gear speed', 'BPFIxPGS +- carrier']]

figures_xlim = [[0, 80], [0, 1.5], [0.5, 0.8], [0, 0.1]

    , [0.5, 0.8], [0.5, 0.8], [0.5, 0.8], [0.5, 0.8]]

figures_ylim = [[0, 0], [0, 40e-7], [0, 6e-7], [0, 0]

    , [0, 6e-7], [0, 6e-7], [0, 6e-7], [0, 6e-7]]

save_files = ['_0maia_500_order_GMF', '_1maia_500_order_BPF', '_8maia_500_order_result',
              '_3maia_500_order_Input_shaft',

              '_4maia_500_order_BPFI', '_5maia_500_order_BPFI',
              '_6maia_500_order_BPFI', '_7maia_500_order_BPFI']
# plt.close('all')
for labels, xlims, ylims, save_str in zip(figures_labels, figures_xlim, figures_ylim, save_files):
    plt.figure()
    plt.plot(freq, sp, label='order spectrum', color='k')
    plt.xlabel('Order [xn]')
    plt.ylabel('PSD')
    if ylims[1] == 0:
        max_sp = max(sp[(freq > xlims[0]) * (freq < xlims[1])])
        # fx = freq[sp == px]
        ylims = [0, 1.1 * max_sp]
    plt.axis([xlims[0], xlims[1], ylims[0], ylims[1]])

    for k, label in enumerate(labels):
        if ' x ' in label:
            label_i = label.split(' x ')
            xcoords = np.r_[all_values_dic[label_i[0]] * all_values_dic[label_i[1]]]
            kpoint = int(freq[-1] / xcoords) + 1
            xcoords = np.r_[1:kpoint + 1] * xcoords

        elif ' +- ' in label:
            label_i = label.split(' +- ')
            xcoords = np.r_[all_values_dic[label_i[0]]]
            kpoint = int(freq[-1] / xcoords) + 1
            xcoords = np.r_[1:kpoint + 1] * xcoords

            xcoords_1 = xcoords + all_values_dic[label_i[1]]
            xcoords_2 = xcoords - all_values_dic[label_i[1]]
            xcoords = np.append(xcoords_1, xcoords_2)
        else:
            xcoords = np.r_[all_values_dic[label]]
            kpoint = int(freq[-1] / xcoords) + 1
            if label == 'GMF 1-2':
                kpoint = 10
            xcoords = np.r_[1:kpoint + 1] * xcoords

        colori = color_s.get(label, 'g')
        plt.vlines(x=xcoords, colors=colori, linestyle=linestyle_str[k], ymin=ylims[0], ymax=ylims[1], label=label,
                   linewidth=1.5)
    plt.grid(True)
    plt.legend(loc='upper right', shadow=True)
    plt.draw()
    save_dir = 'figures/' + save_str + '.pdf'  # f-string
    plt.savefig(save_dir, bbox_inches='tight')

# % Save dataz
sio.savemat('Data/maia_500_angle_domain_speeds.mat',
            all_values_dic, do_compression=True)
plt.show()