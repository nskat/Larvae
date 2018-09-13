'''
This python file implements a routine to use to load larvae data, process it using a layered window Fourier transform
and return the corresponding Fourier components in a Pandas DataFrame
Only kept for information, does not work as well as loader.py
'''

import numpy as np
import pandas as pd
import os
import glob
from math import pi
from time import time
import traceback
import tables
from progress.bar import Bar


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


def load_transform(path, features='position', lines=None, screen='', save_dir='', batch_size=None):
    t0 = time()

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

    if features == 'all':
        feats = ['larva_length_smooth_5', 'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
                 'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                 'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
                 'd_eff_head_norm_deriv_smooth_5','d_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
                 'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5', 'As_smooth_5',
                 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5', 'motion_to_v_tail_head_smooth_5']
    elif features == 'dynamic':
        feats = ['larva_length_deriv_smooth_5', 'S_deriv_smooth_5', 'eig_deriv_smooth_5',
                 'angle_upper_lower_deriv_smooth_5', 'angle_downer_upper_deriv_smooth_5',
                 'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_deriv_smooth_5', 'motion_velocity_norm_smooth_5',
                 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5', 'motion_to_u_tail_head_smooth_5',
                 'motion_to_v_tail_head_smooth_5']
    elif features == 'position':
        feats = ['larva_length_smooth_5', 'S_smooth_5', 'eig_smooth_5', 'angle_upper_lower_smooth_5',
                 'angle_downer_upper_smooth_5', 'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
                 'As_smooth_5', 'prod_scal_1', 'prod_scal_2']

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
    x_shape = (0, len(feats) * (1 + nf) + 1)
    hdf5_path = save_path + '/dataset.hdf5'
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    storage = hdf5_file.create_earray(hdf5_file.root, 'tmp', tables.Float64Atom(), shape=x_shape)
    x_storage = hdf5_file.create_earray(hdf5_file.root, 'x', tables.Float64Atom(), shape=x_shape)

    # Counter
    n_larvae = 0

    # # Initialize the list of lines from the argument passed as a string
    # if screen == 't15':
    #     path = path + '/t15'
    # elif screen == 't5':
    #     path = path + '/t5'
    # else:
    #     raise NotImplementedError

    if lines:
        lines = [x.strip() for x in lines.split(',')]
    else:
        raise NotImplementedError

    # Browse sub folders looking for data
    dirs = [r'/' + dir_ for dir_ in os.listdir(path) for line in lines if line in dir_]
    allFiles_d = {key: [] for key in dirs}
    allFiles = []

    for dir_ in dirs:
        for subdir, _, _ in os.walk(path + dir_):
            allFiles_d[dir_] += glob.glob(subdir + r'/State_Amplitude_t*')

    print('Lines for which data has been found:', [dir_ for dir_ in dirs if allFiles_d[dir_]])

    # Balance dataset by picking the same number of larvae from each line
    smallest_line_len = min([len(i) for i in allFiles_d.values()])
    for dir_ in dirs:
        allFiles += allFiles_d[dir_][:smallest_line_len]

    print(dirs)

    if allFiles:
        # Loading bar
        bar = Bar('Loading files', max=len(allFiles))

        for file_ in allFiles:
            df = pd.read_csv(file_, sep='\t', header=None, names=names)
            Ts = df['t'][1] - df['t'][0]

            # Times of the stimulus are different depending on the screen
            if screen == 't15':
                df = df[((df['t'] > 20) & (df['t'] < 50))]
            elif screen == 't5':
                df = df[((df['t'] > 40) & (df['t'] < 90))]

            if len(df.index) > 250:
                n_larvae += 1
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

                for col in feats:
                    sig = df[col].values
                    tax = np.arange(0, len(sig) * Ts, Ts)
                    tax = tax[:len(sig)]
                    taumax = min(5, int(tax[-1] / 2))
                    tau = np.logspace(np.log10(2 * Ts), np.log10(taumax), 10)  # using 10 points on a logscale

                    try:
                        x.append(LWFT(sig, xtype, tax, nf, sigma, tau, faxtype))
                    except:
                        traceback.print_exc()
                        break
                if np.sum(x) != 0:
                    x = np.hstack((np.vstack(x).T, df[feats].values, df['label'].values[:, None]))
                    storage.append(x)
                    bar.next()
                else:
                    bar.next()

    bar.finish()
    t1 = time()

    np.random.shuffle(hdf5_file.root.tmp[:])

    # If the computation has to be done in batches, resize the array to an even number of batches of the desired size
    if batch_size:
        batch_size = int(batch_size)
        num_max_batches = int(len(hdf5_file.root.tmp[:])/batch_size)
        x_storage.append(hdf5_file.root.tmp[:num_max_batches * batch_size])
        hdf5_file.root.tmp.remove()
    else:
        hdf5_file.root.x.remove()
        hdf5_file.rename_node(where=hdf5_file.root.tmp, newname='x')

    # Shuffle data
    len_dataset = len(hdf5_file.root.x[:])

    hdf5_file.close()

    print("***** Data successfully loaded from ", path, " for a total of ", n_larvae, " larvae samples *****")
    print("***** Import time : ", time() - t0, " *****")
    print("***** Permutation time : ", time() - t1, " *****")

    print('Data saved to', hdf5_path)
    return len_dataset, hdf5_path


def generate_data_ae(dataset, batch_size, len_dataset):
    i = 0
    num_max_batches = int(len_dataset / batch_size)
    while True:
        if (((i + 1) % num_max_batches) < num_max_batches) & (((i + 1) % num_max_batches) > 0):
            batch = dataset.root.x[(i % num_max_batches) * batch_size: ((i + 1) % num_max_batches) * batch_size, :-1]
            i += 1
        else:
            batch = dataset.root.x[(i % num_max_batches) * batch_size:, :-1]
            i = 0
        yield (batch, batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path')
    parser.add_argument('--save_dir')
    parser.add_argument('--lines', help='lines')
    parser.add_argument('--screen')
    parser.add_argument('--batch_size', default=None)
    parser.add_argument('--features', default='position')

    args = parser.parse_args()

    load_transform(args.path, features=args.features, lines=args.lines, save_dir=args.save_dir,
                   screen=args.screen, batch_size=args.batch_size)
