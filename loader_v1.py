'''
This python file implements a routine to use to load larvae data, process it using a layered window Fourier transform
and return the corresponding Fourier components in a Pandas DataFrame
'''

import numpy as np
import pandas as pd
import os
import glob
from math import pi
from time import time
import traceback

def LWFT(x, xtype,tax,nf,sigma,tau,faxtype): # Layered Window Fourier Transform, according to Johnson (2013)
    # xtype = 0 or 1 for forward or inverse transform
    # x = signal
    # tax = time array
    # nf = N in the article
    # faxtype = P in the article

    Ts = tax[1] - tax[0]
    Fs = 1/Ts

    nd = len(tax)       # number of data entries
    tax = tax-tax[0]     # assign 0 to first entry
    nt = (nd - 1)*Ts         # duration of data

    if (faxtype%2) == 1: # case P = 1
    #     fax = np.arange(0, nf + 1)/2/nf                  # frequency axis
        dfax = np.ones(nf+1)/(2*nf*Ts)        # integration measure
        dfax[0] = dfax[0]/2    # edge pixels
        dfax[-1] = dfax[-1]/2 # edge pixels
        fax = np.arange(0, nf + 1)*dfax

    else: # case P = 0
    #     fax = (np.arange(1,nf)-1/2)/2/nf            # frequency axis
        dfax = np.ones(nf)/(2*nf*Ts)        # integration measure
        fax = (2*np.arange(1, nf+1) - 1)*dfax/2

    nff = nf + (faxtype%2)                # number of pixels

    taumax = max(tau)               # maximum window size
    tmax = int(nt + 2*taumax)                 # transform duration
    phi = np.zeros(2*int(np.ceil(taumax/Ts))+1)          # accumulated window
    nwa = 0                            # number of windows accumulated

    for taun in tau:
        nwa = nwa+1                        # next window accumulated
        twn = np.hstack((np.arange(-taun, 0, Ts), np.arange(0, taun + Ts, Ts)))  # window time axis
        phin = np.exp(-pi*(twn**2)/(taun**2)/(sigma**2)/2)  # window amplitude
        phin = phin/np.sqrt(np.dot(phin, phin.T))       # normalized to unit energy
        phi[int((len(phi) - len(phin))/2) : int((len(phi) + len(phin))/2)] += phin**2

    phi = np.sqrt(phi/sum(phi)*2)              # normalized to unit energy x 2
    tw = np.hstack((np.arange(-taumax, 0, Ts), np.arange(0, taumax + Ts, Ts)))    # final window time axis

    if xtype == 1:   # forward transform
        y = np.zeros([nff, nd])
        for kf in range(nff):
            f = fax[kf]
            theta = np.exp(1j*2*pi*f*tw)
            psi = np.multiply(phi, theta)
            rowf = np.convolve(psi, x, mode = 'same')
            y[kf,:] = rowf
    elif xtype == -1:  # inverse transform
        faxtax = np.zeros([nff,nt])
        for kf in range(nff):
            f = fax[kf]
            theta = exp(1j*2*pi*f*tw)
            psi = np.multiply(phi, theta, 'same')
            rowf = np.convolve(psi, x[kf,:])
            faxtax[kf,:] = rowf[taumax*2+ np.arange([1,nt+1])]
        y = np.real(dfax*faxtax[:,tax])
    return y#, fax, tax



def load_transform(PATH, n_samples=10000, features = 'all', mode = 'transforms', labels = 'normal', batch_size=256):
    # Helper function to load the timeseries from all files in subdirectories from PATH,
    # re-scale them, compute their Fourier transforms, and return them in a numpy array,
    # along with the associated labels
    # only time samples between 20s and 55s, then 60s and 90s are considered
    # columns = timeseries from which the LWFT will be computed ; from each column 50 features will be created
    # if 'all' : all features selected, if 'dynamic' only the derivatives, if 'position' only the positions
    # mode allows choose to include in the resulting DataFrame only the Fourier Transforms ('transforms') or
    # also the associated timeseries ('all')
    # labels = types of label selected : 'normal', 'large', 'strong_weak'

    t0 = time()

    if labels == 'normal':
        names = ['t', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll',
               'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba', 'ball_proba',
               'larva_length_smooth_5',
               'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
               'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
               'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
               'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
               'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
               'As_smooth_5', 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5',
               'motion_to_v_tail_head_smooth_5']
        labels_ = ['crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll']

    elif labels == 'large':
        names = ['t','crawl_large', 'bend_large', 'stop_large', 'head retraction_large', 'back crawl_large',
             'roll_large','small_motion', 'straight_proba', 'straight_and_light_bend_proba',
             'bend_proba', 'curl_proba' ,'ball_proba', 'larva_length_smooth_5','larva_length_deriv_smooth_5',
             'S_smooth_5',  'S_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5',
             'angle_upper_lower_deriv_smooth_5', 'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5',
             'd_eff_head_norm_smooth_5',  'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5',
             'd_eff_tail_norm_deriv_smooth_5', 'motion_velocity_norm_smooth_5','head_velocity_norm_smooth_5',
             'tail_velocity_norm_smooth_5', 'As_smooth_5','prod_scal_1' , 'prod_scal_2',
             'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']
        labels_ = ['crawl_large', 'bend_large', 'stop_large', 'head retraction_large', 'back crawl_large', 'roll_large',
                 'small_motion']

    elif labels == 'strong_weak':
        names = ['t','crawl_weak', 'crawl_strong','bend_weak','bend_strong','stop_weak' , 'stop_strong',
             'head retraction weak', 'head retraction strong', 'back crawl weak','back crawl strong',
             'roll weak','roll strong', 'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba' ,
             'ball_proba', 'larva_length_smooth_5','larva_length_deriv_smooth_5', 'S_smooth_5',  'S_deriv_smooth_5',
             'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5','angle_upper_lower_deriv_smooth_5',
             'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
             'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5',  'd_eff_tail_norm_deriv_smooth_5',
             'motion_velocity_norm_smooth_5','head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
             'As_smooth_5','prod_scal_1' , 'prod_scal_2',     'motion_to_u_tail_head_smooth_5',
             'motion_to_v_tail_head_smooth_5']
        labels_ = ['crawl_weak', 'crawl_strong','bend_weak','bend_strong','stop_weak' , 'stop_strong',
             'head retraction weak', 'head retraction strong', 'back crawl weak','back crawl strong',
             'roll weak','roll strong']

    if (features == 'all' or features ==  'cols'):
        feats = [ 'larva_length_smooth_5', 'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
         'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
         'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
         'd_eff_head_norm_deriv_smooth_5','d_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
         'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5', 'As_smooth_5',
         'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']

    elif (features == 'dynamic' or features ==  'cols_dynamic'):
        feats = [ 'larva_length_deriv_smooth_5', 'S_deriv_smooth_5', 'eig_deriv_smooth_5',
         'angle_upper_lower_deriv_smooth_5', 'angle_downer_upper_deriv_smooth_5','d_eff_head_norm_deriv_smooth_5',
         'd_eff_tail_norm_deriv_smooth_5','motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5',
         'tail_velocity_norm_smooth_5']
    elif (features == 'position' or features ==  'cols_positions'):
        feats = [ 'larva_length_smooth_5', 'angle_upper_lower_smooth_5',
         'angle_downer_upper_smooth_5', 'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
         'As_smooth_5', 'motion_to_u_tail_head_smooth_5','motion_to_v_tail_head_smooth_5']
    elif (features == 'position_large' or features == 'cols_positions_large'):
        feats = ['larva_length_smooth_5', 'S_smooth_5',
         'eig_smooth_5', 'angle_upper_lower_smooth_5',
         'angle_downer_upper_smooth_5', 'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
         'As_smooth_5', 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']
    elif (features == 'shape' or features ==  'cols_shape'):
        feats = ['S_smooth_5','eig_smooth_5', 'As_smooth_5','prod_scal_1','prod_scal_2']
    angular_der = ['S_deriv_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                   'angle_downer_upper_deriv_smooth_5']

    data_features = []
    n_larvae = 0
    for dirs, _, _ in os.walk(PATH):
        if labels == 'normal':
            allFiles = glob.glob(dirs + r"\State_Amplitude_t15*.txt")
        elif labels == 'large':
            allFiles = glob.glob(dirs + r"\State_Amplitude_large_state_*.txt")
        elif labels == 'strong_weak':
            allFiles = glob.glob(dirs + r"\State_Amplitude_state_strong_weak*.txt")

        for file_ in allFiles:
            df = pd.read_csv(file_, sep='\t', header=None, names= names)
            Ts = df['t'][1] - df['t'][0]

            df = df[(df['t'] > 25) & (df['t'] < 100)]

            if len(df.index) > 250:
                n_larvae += 1
                # x = np.empty((len(df[labels_[0]].values), 0), int)
                x = []
                # If necessary, removes the imaginary parts
                for col in feats:
                    if df[col].dtype == object:
                        df[col] = (df[col].str.split(('+'))).str[0]
                        df[col] = pd.to_numeric((df[col].str.split(('[0-9]-'))).str[0])

                df[angular_der] = np.tanh((df[angular_der] - df[angular_der].mean()) / (2 * df[angular_der].var()))

                maxs = df[feats].max()
                mins = df[feats].min()
                df[feats] = (df[feats] - mins) / (maxs - mins)

                for i, label in enumerate(labels_):
                    df.loc[df[label] == 1, 'label'] = i

                if ( features == 'cols' or features ==  'cols_dynamic' or features ==  'cols_position' or features=='cols_shape' or features=='cols_position_large'):
                    data_features.append(df[feats])

                elif (features == 'all' or features == 'dynamic' or features == 'position' or features=='shape' or features=='position_large'):
                    for col in feats:
                        sig = df[col].values
                        tax = np.arange(0, len(sig) * Ts, Ts)
                        tax = tax[:len(sig)]

                        xtype = 1  # forward transform
                        nf = 50 # not to make the features space too big
                        sigma = 1  # gaussian windows parameter
                        taumax = min(5, int(tax[-1] / 2))
                        tau = np.logspace(np.log10(2 * Ts), np.log10(taumax), 10)  # using 10 points on a logscale

                        faxtype = 0  # 1 or 0

                        try:
                            # y = LWFT(sig, xtype, tax, nf, sigma, tau, faxtype)
                            # x = np.hstack((x, y.T))
                            x.append(LWFT(sig, xtype, tax, nf, sigma, tau, faxtype))
                        except:
                            traceback.print_exc()
                            break
                    if (mode == 'all') & (np.sum(x) != 0):
                        x = np.hstack((np.vstack(x).T, df[feats].values, df['label'].values[:, None]))
                    if np.sum(x) != 0:
                        data_features.append(x)

    data_features = np.vstack(data_features)

    S_hat = []
    for i in range(len(labels)):
        S_hat.append(data_features[np.where(data_features[:, -1] == i)][:n_samples])

    S_hat = np.vstack(S_hat)

    # Shuffle S_hat
    np.random.shuffle(S_hat)

    # Resize the dataset to be divisible by the batch_size, allowing computations in batches easily
    num_max_batches = int(len(S_hat) / batch_size)
    S_hat = S_hat[:(num_max_batches * batch_size)]

    label = S_hat[:, -1].astype(int)
    S_hat = S_hat[:, :-1]

    print("***** Data successfully loaded from ", PATH, " for a total of ", n_larvae, " larvae samples *****")
    print("***** Import time : ", time() - t0, " *****")

    save_path = "data\data.npz"
    np.savez_compressed(save_path, x=S_hat, y=label)
    print('Data saved to', save_path)

    return S_hat, label


