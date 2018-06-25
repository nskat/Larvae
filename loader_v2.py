'''
This python file implements a routine to use to load larvae data, process it using a layered window Fourier transform
and return the corresponding Fourier components in a Pandas DataFrame
Does not load all files and keep then in memory, but loads them one after the other and saves them in an hdf5 file
'''

import numpy as np
import pandas as pd
import os
import glob
from math import pi
from time import time
import traceback
import tables

def LWFT(x, xtype,tax,nf,sigma,tau,faxtype): # Layered Window Fourier Transform, according to Johnson (2013)
    # xtype = 0 or 1 for forward or inverse transform
    # x = signal
    # tax = time array
    # nf = N in the article
    # faxtype = P in the article

    Ts = tax[1] - tax[0]
    nd = len(tax)                             # number of data entries
    tax = tax-tax[0]                          # assign 0 to first entry
    nt = (nd - 1)*Ts                          # duration of data

    if (faxtype%2) == 1:  # case P = 1
        dfax = np.ones(nf+1)/(2*nf*Ts)        # integration measure
        dfax[0] = dfax[0]/2    # edge pixels
        dfax[-1] = dfax[-1]/2 # edge pixels
        fax = np.arange(0, nf + 1)*dfax       # frequency axis

    else:                 # case P = 0
        dfax = np.ones(nf)/(2*nf*Ts)          # integration measure
        fax = (2*np.arange(1, nf+1) - 1)*dfax/2       # frequency axis

    nff = nf + (faxtype%2)                    # number of pixels
    taumax = max(tau)                         # maximum window size
    phi = np.zeros(2*int(np.ceil(taumax/Ts))+1)       # accumulated window
    nwa = 0                                   # number of windows accumulated

    for taun in tau:
        nwa = nwa+1                           # next window accumulated
        twn = np.hstack((np.arange(-taun, 0, Ts), np.arange(0, taun + Ts, Ts)))  # window time axis
        phin = np.exp(-pi*(twn**2)/(taun**2)/(sigma**2)/2)                       # window amplitude
        phin = phin/np.sqrt(np.dot(phin, phin.T))     # normalized to unit energy
        phi[int((len(phi) - len(phin))/2) : int((len(phi) + len(phin))/2)] += phin**2

    phi = np.sqrt(phi/sum(phi)*2)             # normalized to unit energy x 2
    tw = np.hstack((np.arange(-taumax, 0, Ts), np.arange(0, taumax + Ts, Ts)))   # final window time axis

    if xtype == 1:         # forward transform
        y = np.zeros([nff, nd])
        for kf in range(nff):
            f = fax[kf]
            theta = np.exp(1j*2*pi*f*tw)
            psi = np.multiply(phi, theta)
            rowf = np.convolve(psi, x, mode='same')
            y[kf, :] = rowf
    elif xtype == -1:      # inverse transform
        faxtax = np.zeros([nff,nt])
        for kf in range(nff):
            f = fax[kf]
            theta = exp(1j*2*pi*f*tw)
            psi = np.multiply(phi, theta, 'same')
            rowf = np.convolve(psi, x[kf,:])
            faxtax[kf, :] = rowf[taumax*2 + np.arange([1, nt+1])]
        y = np.real(dfax*faxtax[:, tax])
    return y


def load_transform(path, labels='normal', lines=None, save_dir=''):
    # Helper function to load the timeseries from all files in subdirectories from PATH,
    # re-scale them, compute their Fourier transforms, and return them in a numpy array,
    # along with the associated labels
    # only time samples between 30s and 90s are considered
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
                 'tail_velocity_norm_smooth_5', 'As_smooth_5','prod_scal_1', 'prod_scal_2',
                 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']
        labels_ = ['crawl_large', 'bend_large', 'stop_large', 'head retraction_large', 'back crawl_large', 'roll_large',
                   'small_motion']

    elif labels == 'strong_weak':
        names = ['t','crawl_weak', 'crawl_strong','bend_weak','bend_strong','stop_weak' , 'stop_strong',
                 'head retraction weak', 'head retraction strong', 'back crawl weak','back crawl strong',
                 'roll weak','roll strong', 'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba',
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

    feats = ['larva_length_smooth_5', 'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
             'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
             'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
             'd_eff_head_norm_deriv_smooth_5','d_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
             'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5', 'As_smooth_5',
             'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']

    angular_der = ['S_deriv_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                   'angle_downer_upper_deriv_smooth_5']

    # LWFT parameters
    xtype = 1  # forward transform
    nf = 50  # not to make the features space too big
    sigma = 1  # gaussian windows parameter
    faxtype = 0  # 1 or 0

    save_path = save_dir + '/data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize hdf5 file
    tmp_shape = (0, len(feats)*(1 + nf) + 1)
    hdf5_path_tmp = save_path + '/tmp.hdf5'
    hdf5_tmp = tables.open_file(hdf5_path_tmp, mode='w')
    storage_tmp = hdf5_tmp.create_earray(hdf5_tmp.root, 'tmp', tables.Float64Atom(), shape=tmp_shape)

    # Counters
    n_larvae = 0
    count_labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Initialize the list of lines from the argument passed as a string
    if lines:
        lines = [x.strip() for x in lines.split(',')]
        allFiles = None

    # Browse sub folders looking for data
    for dirs, _, _ in os.walk(path):
        if lines:
            if any(s in dirs for s in lines):
                if labels == 'normal':
                    allFiles = glob.glob(dirs + r"/State_Amplitude_t15*.txt")
                elif labels == 'large':
                    allFiles = glob.glob(dirs + r"/State_Amplitude_large_state_*.txt")
                elif labels == 'strong_weak':
                    allFiles = glob.glob(dirs + r"/State_Amplitude_state_strong_weak*.txt")
        else:
            if labels == 'normal':
                allFiles = glob.glob(dirs + r"/State_Amplitude_t15*.txt")
            elif labels == 'large':
                allFiles = glob.glob(dirs + r"/State_Amplitude_large_state_*.txt")
            elif labels == 'strong_weak':
                allFiles = glob.glob(dirs + r"/State_Amplitude_state_strong_weak*.txt")

        if allFiles:
            for file_ in allFiles:
                df = pd.read_csv(file_, sep='\t', header=None, names=names)
                Ts = df['t'][1] - df['t'][0]

                # We only consider times during and between the stimuli
                df = df[((df['t'] > 25) & (df['t'] < 95))]

                if len(df.index) > 250:
                    n_larvae += 1
                    x = []

                    # If necessary, removes the imaginary parts
                    for col in feats:
                        if df[col].dtype == object:
                            df[col] = (df[col].str.split('+')).str[0]
                            df[col] = pd.to_numeric((df[col].str.split('[0-9]-')).str[0])

                    # Damps derivatives that become too large at some moments
                    df[angular_der] = np.tanh((df[angular_der]-df[angular_der].mean()) / (2*df[angular_der].var()))

                    # Re-scale features
                    maxs = df[feats].max()
                    mins = df[feats].min()
                    df[feats] = (df[feats] - mins) / (maxs - mins)

                    # Add 'label' column
                    for i, label in enumerate(labels_):
                        df.loc[df[label] == 1, 'label'] = i
                        count_labels[i] += len(df[df[label] == 1])

                    # Preprocess each column in the timeseries
                    for col in feats:
                        sig = df[col].values
                        tax = np.arange(0, len(sig) * Ts, Ts)
                        tax = tax[:len(sig)]
                        taumax = min(5, int(tax[-1] / 2))
                        tau = np.logspace(np.log10(2 * Ts), np.log10(taumax), 10)  # using 10 points on a logscale

                        try:
                            x.append(LWFT(sig, xtype, tax, nf, sigma, tau, faxtype))
                        # If we can't compute the LWFT for this column, go to the next timeseries
                        except:
                            break

                    if np.sum(x) != 0:
                        x = np.hstack((np.vstack(x).T, df[feats].values, df['label'].values[:, None]))
                        storage_tmp.append(x)
    t1 = time()

    hdf5_tmp.root.tmp[:].sort()

    n_samples = np.asarray(list(count_labels.values())).min() * 100

    # Initialize hdf5 file
    x_shape = (0, len(feats)*(1 + nf))
    hdf5_path = save_path + '/dataset.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    x_storage = hdf5_file.create_earray(hdf5_file.root, 'x', tables.Float64Atom(), shape=x_shape)
    y_storage = hdf5_file.create_earray(hdf5_file.root, 'y', tables.Int16Atom(), shape=(0,))

    for i in range(len(labels)):
        h = len(hdf5_tmp.root.tmp[np.where(hdf5_tmp.root.tmp[:, -1] == i)[0], :])
        p = np.random.choice(np.arange(h), min(n_samples, h))
        x_storage.append(hdf5_tmp.root.tmp[np.where(hdf5_tmp.root.tmp[:, -1] == i)[0], :][p, :-1])
        y_storage.append(hdf5_tmp.root.tmp[np.where(hdf5_tmp.root.tmp[:, -1] == i)[0], :][p, -1])

    # Shuffle data
    len_dataset = len(hdf5_file.root.x[:])
    p2 = np.random.permutation(len_dataset)
    hdf5_file.root.x[:] = hdf5_file.root.x[p2, :]
    hdf5_file.root.y[:] = hdf5_file.root.y[p2]

    hdf5_tmp.close()
    hdf5_file.close()

    print("***** Data successfully loaded from ", path, " for a total of ", n_larvae, " larvae samples *****")
    print("***** Import time : ", time() - t0, " *****")
    print("***** Permutation time : ", time() - t1, " *****")

    print('Data saved to', hdf5_path)
    return len_dataset, hdf5_path


def generate_data_ae(dataset, batch_size, len_dataset):
    i = 0
    while True:
        if batch_size*(i+1) < len_dataset:
            batch = dataset.root.x[i*batch_size:(i+1)*batch_size]
            i += 1
        else:
            batch = dataset.root.x[i*batch_size:]
            i = 0
        yield (batch, batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path')
    parser.add_argument('--lines', help='lines')

    args = parser.parse_args()

    load_transform(args.path, labels='normal', lines=args.lines)