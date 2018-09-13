'''
Routine to compute time distributions, evolution of probabilities and transition matrices of a model.
Usage :
        python utils/metrics.py --path=/screen --n_clusters=6 --lines=GMR_72F11_AE_01@UAS_Chrimson_Venus_X_0070/r_LED100_30s2x15s30s --window=1 --screen=t15 --idec_weight=/Larvae/window_input_2s_GMR_72F11_FCF_6_clusters
'''

# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"

import os
import numpy as np
from keras.models import Model
import warnings
warnings.filterwarnings("ignore")
from IDEC.DEC import autoencoder
from IDEC.IDEC_GPU import ClusteringLayer
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors


def get_transitions(x, n_clusters=6, idec_weights='', window=1, screen='', path='', tag=''):

    ae, decoder = autoencoder(dims=[x[0].shape[-1] - 2, 500, 500, 2000, 10])
    n_stacks = len([x[0].shape[-1], 500, 500, 2000, 10]) - 1

    hidden = ae.get_layer(name='encoder_%d' % (n_stacks - 1)).output
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(hidden)

    modele = Model(inputs=ae.input, outputs=[clustering_layer, ae.output])

    modele.load_weights(idec_weights)

    transitions_before = np.zeros([n_clusters, n_clusters])
    transitions_stim = np.zeros([n_clusters, n_clusters])

    if screen == 't15':
        Tstim = 30
        Tfin = 45
    elif screen == 't5':
        Tstim = 45
        Tfin = 80

    Ts = 0.08
    window_len = int(np.floor(window/Ts))

    for j, larva in enumerate(x):

        t = larva[:, -1][window_len:-window_len]
        X = larva[:, :-2][window_len:-window_len]

        res = modele.predict(X)
        predictions = res[0].argmax(axis=1)

        if predictions[0] != predictions[1]:
            predictions[0] = predictions[1]

        if predictions[-1] != predictions[-2]:
            predictions[-1] = predictions[-2]

        for i in range(1, len(predictions) - 1):
            if (predictions[i] != predictions[i + 1]) & (predictions[i] != predictions[i - 1]):
                predictions[i] = predictions[i - 1]

        if (max(t) > Tstim) & (min(t) < Tstim):
            idx_stim = np.where(t == max([time for time in t if time < Tstim]))[0][0]
            idx_fin = np.where(t == max([time for time in t if time < Tfin]))[0][0]
            for i in range(idx_stim - 1):
                if predictions[i] != predictions[i + 1]:
                    transitions_before[predictions[i], predictions[i + 1]] += 1
            for i in range(idx_stim, idx_fin - 1):
                if predictions[i] != predictions[i + 1]:
                    transitions_stim[predictions[i], predictions[i + 1]] += 1

    transitions_before /= np.sum(transitions_before, axis=1)[:, None]
    transitions_stim /= np.sum(transitions_stim, axis=1)[:, None]

    save_dir = path + "/" + screen + tag + '/transitions'
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(save_dir + '/transitions_' + str(n_clusters) + '_clusters.npz',
                        x=transitions_before, t=transitions_stim)

    # Save plot as png
    plt.matshow(transitions_before)
    plt.title('Transitions before stimulus, ' + str(n_clusters) + ' clusters')
    plt.savefig(save_dir + '/transitions_before_stimulus_' + str(tag[1:]) + '_' + str(n_clusters) + '_clusters' '.png')

    plt.matshow(transitions_stim)
    plt.title('Transitions during stimulus, ' + str(n_clusters) + ' clusters')
    plt.savefig(save_dir + '/transitions_during_stimulus_' + str(tag[1:]) + '_' + str(n_clusters) + '_clusters' '.png')
    print('Successfully computed transitions matrices')



def get_distributions(x, n_clusters=6, idec_weights='', window=1, screen='', path='', tag=''):

        ae, decoder = autoencoder(dims=[x[0].shape[-1 ] -2, 500, 500, 2000, 10])
        n_stacks = len([x[0].shape[-1 ] -2, 500, 500, 2000, 10]) - 1

        hidden = ae.get_layer(name='encoder_%d' % (n_stacks - 1)).output
        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(hidden)

        modele = Model(inputs=ae.input, outputs=[clustering_layer, ae.output])

        modele.load_weights(idec_weights)

        analysis = []
        duration_clustering = [[] for i in range(n_clusters)]

        Ts = 0.08
        window_len = int(window/Ts)
        for j, larva in enumerate(x):

            t = larva[:, -1][window_len:-window_len]
            X = larva[:, :-2][window_len:-window_len]

            res = modele.predict(X)
            predictions = res[0].argmax(axis = 1)

            if predictions[0] != predictions[1]:
                predictions[0] = predictions[1]

            if predictions[-1] != predictions[-2]:
                predictions[-1] = predictions[-2]

            for i in range(1, len(predictions) - 1):
                if(predictions[i] != predictions[i + 1]) & (predictions[i] != predictions[i - 1]):
                    predictions[i] = predictions[i - 1]

            predictions = pd.DataFrame(predictions)

            for i in range(n_clusters):
                indices = list(predictions[predictions[0] == i].index)
                if len(indices) > 0:
                    indices_change = [indices[i] + 1 for i in range(len(indices) - 1) if (indices[i] + 1 != indices[i + 1]) or (indices[i - 1] + 1 != indices[i])] + \
                                         [indices[-1] + 1]
                    times_change = [t[i - 1] for i in indices_change]
                    times_change = [[times_change[i], times_change[i + 1]] for i in range(len(times_change)) if
                                    (i % 2 == 0)]
                    len_behavior = np.array([sublist[1] - sublist[0] for sublist in times_change]).flatten().tolist()

                    if len(len_behavior) > 0:
                        duration_clustering[i] += len_behavior

            analysis.append([predictions.values, t])

        save_dir = path + "/" + screen + tag + '/distributions'
        print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez_compressed(save_dir + '/distributions_' + str(n_clusters) + '_clusters.npz',
                            x=np.array(analysis), t=np.array(duration_clustering))

        # Save plot as png
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        bounds = [i for i in range(len(duration_clustering) + 1)]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(duration_clustering))
        for i in range(len(duration_clustering)):
            Y, X = np.histogram(duration_clustering[i], 100, density=True, range=(0, 4))
            cm = plt.cm.get_cmap('Paired')
            h = plt.bar(X[:-1], Y, width=X[1] - X[0], color=cm(i))
            plt.title('Time distributions ' + tag[1:] + ' ' + str(n_clusters) + ' clusters')

        ax2 = fig.add_axes([0.95, 0.12, 0.03, 0.7])
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=bounds,
                                        spacing='proportional')
        plt.savefig(save_dir + '/time_distributions' + str(tag[1:]) + '_' + str(n_clusters) + '_clusters' + '.png')
        print('Successfully computed distributions')


def get_probas(path='', idec_weights='', n_clusters=6, window=1, screen='t15', tag=''):
    n_clusters = int(n_clusters)
    window = int(window)

    columns = ['t', 'crawl', 'bend', 'stop', 'head retraction', 'back crawl', 'roll',
                 'straight_proba', 'straight_and_light_bend_proba', 'bend_proba', 'curl_proba', 'ball_proba',
                 'larva_length_smooth_5',
                 'larva_length_deriv_smooth_5', 'S_smooth_5', 'S_deriv_smooth_5',
                 'eig_smooth_5', 'eig_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                 'angle_downer_upper_smooth_5', 'angle_downer_upper_deriv_smooth_5', 'd_eff_head_norm_smooth_5',
                 'd_eff_head_norm_deriv_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
                 'motion_velocity_norm_smooth_5', 'head_velocity_norm_smooth_5', 'tail_velocity_norm_smooth_5',
                 'As_smooth_5', 'prod_scal_1', 'prod_scal_2', 'motion_to_u_tail_head_smooth_5',
                 'motion_to_v_tail_head_smooth_5']

    feats = ['larva_length_smooth_5', 'S_smooth_5', 'eig_smooth_5', 'angle_upper_lower_smooth_5',
                 'angle_downer_upper_smooth_5',  'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
                 'As_smooth_5', 'prod_scal_1', 'prod_scal_2']

    if screen == 't15':
        start_time = 20
        end_time = 60
    elif screen == 't5':
        start_time = 40
        end_time = 95

    Ts = 0.08

    window_len = int(np.floor(float(window) / Ts))
    x_shape = (0, (2*window_len + 1)*len(feats))

    ae, decoder = autoencoder(dims=[x_shape[-1], 500, 500, 2000, 10])
    n_stacks = len([x_shape[-1], 500, 500, 2000, 10]) - 1

    hidden = ae.get_layer(name='encoder_%d' % (n_stacks - 1)).output
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(hidden)

    modele = Model(inputs=ae.input, outputs=[clustering_layer, ae.output])

    modele.load_weights(idec_weights)

    files = []
    proba = np.zeros((int((end_time - start_time)/Ts), n_clusters))

    for dirs, _, _ in os.walk(path):
        files += glob.glob(dirs + r"/State_Amplitude_t*.txt")

    if files:
        for file in sorted(files):

            df = pd.read_csv(file, sep='\t', header=None, names=columns)

            # Reducing the length of df to the moments of interest. We shorten it to length (end_time - start_time)/Ts + 2*window_len b/c of sampling discrepancies
            df = df[(df['t'] > (start_time - window)) & (df['t'] < (end_time + window))].reset_index(drop=True)[: int(len(proba) + 2*window_len)]

            for col in feats:
                if df[col].dtype == object:
                    try:
                        df[col] = (df[col].str.split('+')).str[0]
                        df[col] = pd.to_numeric((df[col].str.split('[0-9]-')).str[0])
                    except:
                        break

            # Re-scale features
            maxs = df[feats].max()
            mins = df[feats].min()
            df[feats] = (df[feats] - mins) / (maxs - mins)

            if len(df) > 100:
                x = []
                for i in range(window_len, len(df) - window_len):
                    if not (np.isnan(df[feats][i - window_len:i + window_len + 1].T.values.flatten()).any()):
                        x.append(np.array(df[feats][i - window_len:i + window_len + 1].T.values.flatten())[:, None].T)
                x = np.vstack(x)

                time = df[(df['t'] > start_time) & (df['t'] < end_time + 1)]['t'].values[:len(x)]

                if not(len(x) <= len(proba)) or not(len(x) == len(time)):
                    continue

                res = modele.predict(x)
                predictions = res[0].argmax(axis=1)

                for i in range(len(x)):
                    time_index = min(int((time[i] - start_time) / Ts) - 1, len(proba) - 1)
                    proba[time_index, predictions[i]] += 1
        proba = proba / np.sum(proba, axis=1)[:, None]
        times = np.arange(start_time, end_time, Ts)[:len(proba)]

        save_dir = path + "/" + screen + tag + '/probabilities'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.savez_compressed(save_dir + '/probas_' + str(n_clusters) + '_clusters.npz', x=proba, t=times)


        # Save plot as png
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        bounds = [i for i in range(proba.shape[-1] + 1)]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=proba.shape[-1])
        cm = plt.cm.get_cmap('Paired')

        for i in range(proba.shape[-1]):
            plt.plot(times, proba[:, i], color=cm(i))
            plt.title('Probabilities ' + tag[1:] + ' ' + n_clusters + ' clusters')

        ax2 = fig.add_axes([0.95, 0.12, 0.03, 0.7])
        cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cm,
                                        norm=norm,
                                        boundaries=bounds,
                                        ticks=bounds,
                                        spacing='proportional')
        plt.savefig(save_dir + '/probabilities' + str(tag[1:]) + '_' + str(n_clusters) + '_clusters' '.png')
        print('Successfully computed probabilities')


def load_transform(path, window=1, screen='', lines=''):

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

        feats = ['larva_length_smooth_5', 'S_smooth_5', 'eig_smooth_5', 'angle_upper_lower_smooth_5',
                 'angle_downer_upper_smooth_5', 'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
                 'As_smooth_5', 'prod_scal_1', 'prod_scal_2']

        data = []
        n_larvae = 0
        # Initialize the list of lines from the argument passed as a string
        if screen == 't15':
            path = path + '/t15'
        elif screen == 't5':
            path = path + '/t5'
        else:
            raise NotImplementedError

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

        if allFiles:
            for file_ in allFiles:
                df = pd.read_csv(file_, sep='\t', header=None, names=names)

                for i, label in enumerate(labels_):
                    df.loc[df[label] == 1, 'label'] = i

                df = df.reset_index(drop=True)
                n_larvae += 1

                # If necessary, removes the imaginary parts
                for col in feats:
                    if df[col].dtype == object:
                        df[col] = (df[col].str.split(('+'))).str[0]
                        df[col] = pd.to_numeric((df[col].str.split(('[0-9]-'))).str[0])

                    # re-scaling the selected features
                    maxs = df[feats].max()
                    mins = df[feats].min()
                    df[feats] = (df[feats] - mins) / (maxs - mins)

                x = []
                Ts = 0.08
                window_len = int(np.floor(float(window) / Ts))
                # Pad data with zeros at the beginning
                x.append(np.zeros([window_len, (2 * window_len + 1) * len(feats) + 2]))

                for i in range(window_len, len(df) - window_len):
                    if not (np.isnan(df[feats][i - window_len:i + window_len + 1].T.values.flatten()).any()):
                        x.append(
                            np.hstack((df[feats][i - window_len:i + window_len + 1].T.values.flatten(),
                                       df['label'][i],
                                       df['t'][i]))[:, None].T)
                # Pad data with zeros at the end
                x.append(np.zeros([window_len, (2 * window_len + 1) * len(feats) + 2]))
                x = np.vstack(x)
                data.append(x)
        tag = dirs[0]
        print(tag)
        print("***** Data successfully loaded from ", path, " *****")
        return data, tag


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path')
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--lines', help='lines')
    parser.add_argument('--window', type=int)
    parser.add_argument('--screen')
    parser.add_argument('--idec_weights', default=None)

    args = parser.parse_args()

    x, tag = load_transform(path=args.path, lines=args.lines, window=args.window, screen=args.screen)
    get_transitions(x=x, n_clusters=args.n_clusters, idec_weights=args.idec_weights,
                    screen=args.screen, window=args.window, path=args.path, tag=tag)
    get_distributions(x=x, n_clusters=args.n_clusters, idec_weights=args.idec_weights, screen=args.screen,
                      window=args.window, path=args.path, tag=tag)
    get_probas(path=args.path, idec_weights=args.idec_weights, n_clusters=args.n_clusters, window=args.window,
               screen=args.screen, tag=tag)
