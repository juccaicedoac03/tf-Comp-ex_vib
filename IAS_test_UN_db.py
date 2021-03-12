import os

#if os.getcwd() != r'G:\Mon Drive\GCPDS\2019\Python Scripts\STSK\new_window_2a8':
#    print('Changing Dir...')
#    os.chdir('G:/Mon Drive/GCPDS/2019/Python Scripts/STSK/new_window_2a8')
#print(f'Current Dir=\n{os.getcwd()}')
import scipy.io as sio  # load save matlab files
import pandas as pd
#import matplotlib.pyplot as plt  # plotting
import numpy as np
import scipy.signal as sg
import scipy.interpolate as interpolate
import pickle
import scipy.spatial.distance as dis
from scipy.signal import stft


#from functions.STSK import COT_intp2, PSD_envW, SK_W

def bayes_entropy2(X,r,tau):
    pos = 0
    Xm = []
    while pos+tau+1<X.shape[1]:
        Xm.append(X[:,pos:pos+tau])
        pos = pos+tau
    D = []
    for sim in Xm:
        if len(D)==0:
            D.append(sim.ravel())
        elif np.sum(dis.cdist(np.asarray(D),sim)<r)==False:
            D.append(sim.ravel())
    Xm = [i.ravel() for i in Xm]
    xx = np.asarray(Xm)
    d = dis.cdist(np.asarray(D),np.asarray(Xm))
     sig = np.median(d)
    simil = np.exp(-((d)**2)/(2*sig**2))
    #print(sum(simil.ravel()))
    Pd = np.mean(simil,axis=1)
    m  = np.mean(np.asarray([xx[:,i].reshape(xx[:,i].shape[0],1)*simil.T for i in range(xx.shape[1])]),axis=1).T
    var = np.mean(((dis.cdist(xx,m).T)**2)*simil,axis=1)
    d2 = dis.cdist(xx,m)
    simil2 = np.exp(-((d2)**2)/(2*var))
    Psd = np.mean(simil2,axis=0)
    Pds = Psd*Pd
    #print(len(D),len(Xm))
    E = -np.sum(Pds*np.log(Pds))/len(Xm)
    return E

def PSD_envW(x,nfft,Noverlap,Window,Nwind2,nfft2,filterr,fs):
    Window = Window.reshape(-1,1)[:,0]
    n = len(x)          # Number of data points
    nwind = len(Window) # length of window
    x = x.reshape(-1,1)[:,0]
    K = int(np.fix((n-Noverlap)/(nwind-Noverlap))) # Number of windows
    Noverlap2 = int(np.round(3/4*Nwind2))
    Window2 = np.hanning(Nwind2)

    # compute 
    index = np.r_[:nwind]
    #t = np.r_[:n]
    #print(n,nwind,K,Noverlap2,Window,Window2.shape)
    psd = np.zeros(nfft)
    SK_w = np.zeros([nfft2,K])
    for i in np.r_[:K]:
        xw = Window*x[index]
        # filtering
        if filterr == 1:
            f,t,Z = stft(xw.ravel(),fs,nperseg=Noverlap2,nfft=nfft2)
            #print(Z.shape)
            Xz = np.abs(Z)
            entropy = np.asarray([bayes_entropy2(Xz[i,:].reshape(1,-1),np.mean(np.std(Xz,axis=1)),3) for i in range(Xz.shape[0])])[0:-1]
            #print(entropy.shape)
            SK = np.concatenate((entropy,entropy[::-1]),axis=0)
            #SK,_,_,_ = SK_W(xw,nfft2,Noverlap2,Window2)
            SK_w[:,i] = SK.ravel()
            b = np.fft.fftshift(np.real(np.fft.ifft(SK)))
            xw = sg.fftconvolve(xw,b,mode='same') #xw = fftfilt(b,xw);
        xw = np.abs(sg.hilbert(xw)) # envelope
        xw = xw**2
        xw = xw - np.mean(xw)
        xw = sg.hilbert(xw)
        Xw = np.fft.fft(xw,nfft)/nfft
        psd = np.abs(Xw)**2 + psd
        index = index + (nwind - Noverlap
    # normalize
    KMU = K*np.linalg.norm(Window)**2;  #Normalizing scale factor ==> asymptotically unbiased
    psd = psd/KMU
 
    freq = np.fft.fftfreq(psd.shape[-1])
    psd = psd[:int(len(psd)/2)]
    freq = freq[:int(len(freq)/2)]
    return psd,freq,K,SK_w


#filed = 'Data/names_test_UN'
#mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
#names = mat_contents['names']
#df = pd.DataFrame(names, columns=['record_name'])
#df.to_csv('Data/text_UN_names.csv', index=False)
df = pd.read_csv('Data/text_UN_labels.csv', delimiter= ';') # del-4 err-12
cnt1 = -1
cnt2 = 0
X_sk = np.zeros([42-4, 256+1])
X_stat = np.zeros([42-4, 256*2+1])
label = list()
for record in df['record_name']:
    cnt1 += 1
    record = record.strip()
    filed = 'Data/'+record
    mat_contents = sio.loadmat(filed, mdict=None, appendmat=True)
    #f_est = mat_contents['f_est'].ravel()
    fs = mat_contents['fs'].ravel()
    #t = np.r_[:len(f_est)]/25.6e3
    #plt.figure()
    #plt.plot(t, f_est)
    #plt.title(record)
    #plt.show()

    datas = mat_contents['data_s'].ravel()
    FTF = 0.3913
    Nw1 = int(2 ** np.ceil(np.log2(1 / FTF * fs)) * 2 ** 3)
    Nw2: int = 2 ** 8
    Nfft2 = 2 * Nw2
    Noverlap2 = round(3 / 4 * Nw2)
    Nfft = 2**13
    Noverlap = round(3 / 4 * Nw1)
    Window = np.kaiser(Nw1, beta=0)
    if df['error'][cnt1] == 'del':
        print('deleting...')
    if df['error'][cnt1] != 'del':
        #SK, _, _, _ = SK_W(datas, Nfft2, Noverlap2, Nw2)
        f,t,Z = stft(datas,fs,nperseg=Nw,nfft=Nfft2)
        Xz = np.abs(Z)
        entropy = np.asarray([bayes_entropy2(Xz[i,:].reshape(1,-1),np.mean(np.std(Xz,axis=1)),3) for i in range(Xz.shape[0])])

        #sp, freq, K, Sk_w = PSD_envW(datas, Nfft, Noverlap, Window, Nw2, Nfft2, 1)
        psd,f,K,E_w = PSD_envW(datas,Nfft,Noverlap,Window,Nw2,Nfft2,filterr,fs)
        label.append(df['label'][cnt1])
        X_sk[cnt2, :256] = SK[:256]
        X_stat[cnt2, :256] = np.mean(Sk_w, axis=1)[:256]
        X_stat[cnt2, 256:-1] = np.std(Sk_w, axis=1)[:256]
        cnt2 += 1

    results = {}
    results['entropy'] = entropy
    results['entropy_w'] = E_w
    results['K'] = K
    results['f'] = f
    results['psd'] = psd
    with open('Results_UN/'+record+'.p','wb') as handle:
                pickle.dump(results,handle)

X_sk[:,-1] = label
X_stat [:, -1] = label
result_dict = {'X_sk': X_sk, 'X_stat': X_stat}
savename = 'Test_UN_db'
with open('Data/' + savename + '.p','wb') as handle:
                pickle.dump(result_dict,handle)

