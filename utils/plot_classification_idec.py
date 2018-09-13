# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"


import pandas as pd
import scipy.io as sio
from natsort import natsorted
import os
import glob
import numpy as np
from keras.models import Model

import warnings
warnings.filterwarnings("ignore")
import tables
from IDEC.DEC import autoencoder
from IDEC.IDEC_GPU import ClusteringLayer


def create_trx(path, n_clusters=6, weights='', window=1):
    try:
        trx_ = sio.loadmat(path + "/trx.mat")
        trx = trx_['trx']

        data_types = (
                     'full_path', 'id', 'numero_larva', 'numero_larva_num', 'protocol', 'pipeline', 'stimuli', 'neuron',
                     't', 'x_spine', 'y_spine', 'x_contour', 'y_contour', 'x_center', 'y_center', 'straight_proba',
                     'bend_proba',
                     'curl_proba', 'ball_proba', 'straight_and_light_bend_proba', 'global_state', 'x_neck_down',
                     'y_neck_down', 'x_neck_top',
                     'y_neck_top', 'x_neck', 'y_neck', 'x_head', 'y_head', 'x_tail', 'y_tail', 'S', 'prod_scal_1',
                     'prod_scal_2', 'S_smooth_5',
                     'S_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                     'angle_downer_upper_smooth_5',
                     'angle_downer_upper_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5',
                     'head_velocity_norm_smooth_5',
                     'tail_velocity_norm_smooth_5', 'motion_velocity_norm_smooth_5', 'motion_to_u_tail_head_smooth_5',
                     'motion_to_v_tail_head_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
                     'd_eff_head_norm_smooth_5', 'd_eff_head_norm_deriv_smooth_5', 'larva_length_smooth_5',
                     'larva_length_deriv_smooth_5',
                     'proba_global_state', 'run', 'cast', 'stop', 'hunch', 'back', 'roll', 'small_motion', 'start_stop',
                     't_start_stop',
                     'n_duration', 'nb_action', 'As_smooth_5', 'global_state_large_state',
                     'global_state_small_large_state', 'start_stop_large',
                     't_start_stop_large', 'duration_large', 'n_duration_large', 'nb_action_large',
                     'start_stop_large_small',
                     't_start_stop_large_small', 'duration_large_small', 'n_duration_large_small',
                     'nb_action_large_small', 'run_large',
                     'cast_large', 'stop_large', 'hunch_large', 'back_large', 'roll_large', 'run_weak', 'cast_weak',
                     'stop_weak',
                     'hunch_weak', 'back_weak', 'roll_weak', 'run_strong', 'cast_strong', 'stop_strong', 'hunch_strong',
                     'back_strong',
                     'roll_strong', 'global_state_clustering', 'start_stop_clustering', 't_start_stop_clustering',
                     'duration_clustering',
                     'n_duration_clustering', 'nb_action_clustering') + tuple(
            'clustering_' + str(i) for i in range(n_clusters))

        x = load_transform(path, window=window)

        trx_new = []

        ae, decoder = autoencoder(dims=[x[0].shape[-1] - 2, 500, 500, 2000, 10])
        n_stacks = len([x[0].shape[-1] - 2, 500, 500, 2000, 10]) - 1

        hidden = ae.get_layer(name='encoder_%d' % (n_stacks - 1)).output
        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(hidden)

        modele = Model(inputs=ae.input, outputs=[clustering_layer, ae.output])

        modele.load_weights(weights)

        for j, larva in enumerate(x):
            t = larva[:, -1]

            X = larva[:, :-2]

            res = modele.predict(X)
            predictions = res[0].argmax(axis=1)

            if predictions[0] != predictions[1]:
                predictions[0] = predictions[1]

            if predictions[-1] != predictions[-2]:
                predictions[-1] = predictions[-2]

            for i in range(1, len(predictions) - 1):
                if (predictions[i] != predictions[i + 1]) & (predictions[i] != predictions[i - 1]):
                    predictions[i] = predictions[i - 1]

            predictions = pd.DataFrame(predictions)

            global_state_clustering = np.array([i + 1 for i in predictions.values])
            start_stop_clustering = [[] for i in range(n_clusters)]
            t_start_stop_clustering = [[] for i in range(n_clusters)]
            duration_clustering = [[] for i in range(n_clusters)]
            n_duration_clustering = [[] for i in range(n_clusters)]
            nb_action_clustering = [[] for i in range(n_clusters)]
            states = []

            for i in range(n_clusters):
                indices = list(predictions[predictions[0] == i].index)

                if len(indices) > 0:
                    indices_change = [indices[i] + 1 for i in range(len(indices) - 1) if
                                      (indices[i] + 1 != indices[i + 1]) or (indices[i - 1] + 1 != indices[i])] + [
                                         indices[-1] + 1]
                    times_change = [t[i - 1] for i in indices_change]
                    indices_change = [[indices_change[i], indices_change[i + 1]] for i in range(len(indices_change)) if
                                      (i % 2 == 0)]
                    times_change = [[times_change[i], times_change[i + 1]] for i in range(len(times_change)) if
                                    (i % 2 == 0)]
                    len_behavior = [[sublist[1] - sublist[0]] for sublist in times_change]
                    n_duration = [[sublist[1] - sublist[0]] for sublist in indices_change]
                    nb_action = len(indices_change)

                    start_stop_clustering[i] = indices_change
                    t_start_stop_clustering[i] = times_change
                    duration_clustering[i] = len_behavior
                    n_duration_clustering[i] = n_duration
                    nb_action_clustering[i] = nb_action

                states.append([[1] if predictions.values[j] == i else [-1] for j in range(len(predictions))])

            start_stop_clustering = np.array([[i for i in start_stop_clustering]])
            t_start_stop_clustering = np.array([[i for i in t_start_stop_clustering]])
            duration_clustering = np.array([[i if (len(i) > 0) else [[0]] for i in duration_clustering]])
            n_duration_clustering = np.array([i if (len(i) > 0) else [[0]] for i in n_duration_clustering])
            nb_action_clustering = np.array(
                [[np.array([i]).astype('O') if (i) else np.array([0]).astype('O') for i in nb_action_clustering]])

            tmp = [global_state_clustering, start_stop_clustering, t_start_stop_clustering, duration_clustering,
                   n_duration_clustering, nb_action_clustering]
            tmp += states
            tmp = tuple(tmp)

            trx_new.append(np.reshape(
                np.array(tuple(trx[j][0]) + tmp, dtype=[(n, d) for (n, d) in zip(data_types, ["O"] * len(data_types))]),
                [1, ]))

        trx_new = np.array(trx_new)
        trx_['trx'] = trx_new
        sio.savemat(path + '/trx_new_%d_clusters' % n_clusters, trx_, long_field_names=True)
    except NotImplementedError:
        trx_ = tables.open_file(path + "trx.mat")

        data_titles_old = (
        'full_path', 'id', 'numero_larva', 'numero_larva_num', 'protocol', 'pipeline', 'stimuli', 'neuron',
        't', 'x_spine', 'y_spine', 'x_contour', 'y_contour', 'x_center', 'y_center', 'straight_proba', 'bend_proba',
        'curl_proba', 'ball_proba', 'straight_and_light_bend_proba', 'global_state', 'x_neck_down', 'y_neck_down',
        'x_neck_top',
        'y_neck_top', 'x_neck', 'y_neck', 'x_head', 'y_head', 'x_tail', 'y_tail', 'S', 'prod_scal_1', 'prod_scal_2',
        'S_smooth_5',
        'S_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
        'angle_downer_upper_smooth_5',
        'angle_downer_upper_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5', 'head_velocity_norm_smooth_5',
        'tail_velocity_norm_smooth_5', 'motion_velocity_norm_smooth_5', 'motion_to_u_tail_head_smooth_5',
        'motion_to_v_tail_head_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
        'd_eff_head_norm_smooth_5', 'd_eff_head_norm_deriv_smooth_5', 'larva_length_smooth_5',
        'larva_length_deriv_smooth_5',
        'proba_global_state', 'run', 'cast', 'stop', 'hunch', 'back', 'roll', 'small_motion', 'start_stop',
        't_start_stop',
        'n_duration', 'nb_action', 'As_smooth_5', 'global_state_large_state', 'global_state_small_large_state',
        'start_stop_large',
        't_start_stop_large', 'duration_large', 'n_duration_large', 'nb_action_large', 'start_stop_large_small',
        't_start_stop_large_small', 'duration_large_small', 'n_duration_large_small', 'nb_action_large_small',
        'run_large',
        'cast_large', 'stop_large', 'hunch_large', 'back_large', 'roll_large', 'run_weak', 'cast_weak', 'stop_weak',
        'hunch_weak', 'back_weak', 'roll_weak', 'run_strong', 'cast_strong', 'stop_strong', 'hunch_strong',
        'back_strong',
        'roll_strong')
        data_titles = ('full_path', 'id', 'numero_larva', 'numero_larva_num', 'protocol', 'pipeline', 'stimuli',
                       'neuron',
                       't', 'x_spine', 'y_spine', 'x_contour', 'y_contour', 'x_center', 'y_center', 'straight_proba',
                       'bend_proba',
                       'curl_proba', 'ball_proba', 'straight_and_light_bend_proba', 'global_state', 'x_neck_down',
                       'y_neck_down', 'x_neck_top',
                       'y_neck_top', 'x_neck', 'y_neck', 'x_head', 'y_head', 'x_tail', 'y_tail', 'S', 'prod_scal_1',
                       'prod_scal_2', 'S_smooth_5',
                       'S_deriv_smooth_5', 'angle_upper_lower_smooth_5', 'angle_upper_lower_deriv_smooth_5',
                       'angle_downer_upper_smooth_5',
                       'angle_downer_upper_deriv_smooth_5', 'eig_smooth_5', 'eig_deriv_smooth_5',
                       'head_velocity_norm_smooth_5',
                       'tail_velocity_norm_smooth_5', 'motion_velocity_norm_smooth_5', 'motion_to_u_tail_head_smooth_5',
                       'motion_to_v_tail_head_smooth_5', 'd_eff_tail_norm_smooth_5', 'd_eff_tail_norm_deriv_smooth_5',
                       'd_eff_head_norm_smooth_5', 'd_eff_head_norm_deriv_smooth_5', 'larva_length_smooth_5',
                       'larva_length_deriv_smooth_5',
                       'proba_global_state', 'run', 'cast', 'stop', 'hunch', 'back', 'roll', 'small_motion',
                       'start_stop', 't_start_stop',
                       'n_duration', 'nb_action', 'As_smooth_5', 'global_state_large_state',
                       'global_state_small_large_state', 'start_stop_large',
                       't_start_stop_large', 'duration_large', 'n_duration_large', 'nb_action_large',
                       'start_stop_large_small',
                       't_start_stop_large_small', 'duration_large_small', 'n_duration_large_small',
                       'nb_action_large_small', 'run_large',
                       'cast_large', 'stop_large', 'hunch_large', 'back_large', 'roll_large', 'run_weak', 'cast_weak',
                       'stop_weak',
                       'hunch_weak', 'back_weak', 'roll_weak', 'run_strong', 'cast_strong', 'stop_strong',
                       'hunch_strong', 'back_strong',
                       'roll_strong', 'global_state_clustering', 'start_stop_clustering', 't_start_stop_clustering',
                       'duration_clustering',
                       'n_duration_clustering', 'nb_action_clustering') + tuple(
            'clustering_' + str(i) for i in range(n_clusters))

        x = load_transform(path, window=window)
        trx_new = []

        ae, decoder = autoencoder(dims=[x[0].shape[-1] - 2, 500, 500, 2000, 10])
        n_stacks = len([x[0].shape[-1] - 2, 500, 500, 2000, 10]) - 1

        hidden = ae.get_layer(name='encoder_%d' % (n_stacks - 1)).output
        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(hidden)

        modele = Model(inputs=ae.input, outputs=[clustering_layer, ae.output])

        modele.load_weights(weights)

        for j, larva in enumerate(x):
            t = larva[:, -1]

            X = larva[:, :-2]

            res = modele.predict(X)
            predictions = res[0].argmax(axis=1)

            if predictions[0] != predictions[1]:
                predictions[0] = predictions[1]

            if predictions[-1] != predictions[-2]:
                predictions[-1] = predictions[-2]

            for i in range(1, len(predictions) - 1):
                if (predictions[i] != predictions[i + 1]) & (predictions[i] != predictions[i - 1]):
                    predictions[i] = predictions[i - 1]

            predictions = pd.DataFrame(predictions)

            global_state_clustering = np.array([i + 1 for i in predictions.values])
            start_stop_clustering = [[] for i in range(n_clusters)]
            t_start_stop_clustering = [[] for i in range(n_clusters)]
            duration_clustering = [[] for i in range(n_clusters)]
            n_duration_clustering = [[] for i in range(n_clusters)]
            nb_action_clustering = [[] for i in range(n_clusters)]
            states = []

            for i in range(n_clusters):
                indices = list(predictions[predictions[0] == i].index)

                if len(indices) > 0:
                    indices_change = [indices[i] + 1 for i in range(len(indices) - 1) if
                                      (indices[i] + 1 != indices[i + 1]) or (indices[i - 1] + 1 != indices[i])] + [
                                         indices[-1] + 1]
                    times_change = [t[i - 1] for i in indices_change]
                    indices_change = [[indices_change[i], indices_change[i + 1]] for i in range(len(indices_change)) if
                                      (i % 2 == 0)]
                    times_change = [[times_change[i], times_change[i + 1]] for i in range(len(times_change)) if
                                    (i % 2 == 0)]
                    len_behavior = [[sublist[1] - sublist[0]] for sublist in times_change]
                    n_duration = [[sublist[1] - sublist[0]] for sublist in indices_change]
                    nb_action = len(indices_change)

                    start_stop_clustering[i] = indices_change
                    t_start_stop_clustering[i] = times_change
                    duration_clustering[i] = len_behavior
                    n_duration_clustering[i] = n_duration
                    nb_action_clustering[i] = nb_action

                states.append([[1] if predictions.values[j] == i else [-1] for j in range(len(predictions))])

            start_stop_clustering = np.array([[i for i in start_stop_clustering]])
            t_start_stop_clustering = np.array([[i for i in t_start_stop_clustering]])
            duration_clustering = np.array([[i if (len(i) > 0) else [[0]] for i in duration_clustering]])
            n_duration_clustering = np.array([i if (len(i) > 0) else [[0]] for i in n_duration_clustering])
            nb_action_clustering = np.array(
                [[np.array([i]).astype('O') if (i) else np.array([0]).astype('O') for i in nb_action_clustering]])

            tmp = [global_state_clustering, start_stop_clustering, t_start_stop_clustering, duration_clustering,
                   n_duration_clustering, nb_action_clustering]
            tmp += states
            tmp = tuple(tmp)

            trx_new.append(np.reshape(np.array(tuple(trx_.root.trx[k][0][j][0].T for k in data_titles_old) + tmp,
                                               dtype=[(n, d) for (n, d) in zip(data_titles, ["O"] * len(data_titles))]),
                                      [1, ]))

        trx_new = np.array(trx_new, ndmin=2)
        trx_new = {'__header__': '', '__version__': '', 'trx': trx_new}
        sio.savemat(path + '/trx_new_%d_clusters' % n_clusters, trx_new, long_field_names=True)
        trx_.close()


def load_transform(path, window=1):

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

    data_features = []
    n_larvae = 0
    for dirs, _, _ in os.walk(path):
        allFiles = glob.glob(dirs + r"\State_Amplitude_t*.txt")

        for file_ in natsorted(allFiles):
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
            x.append(np.zeros([window_len, (2 * window_len + 1) * len(feats) + 2]))

            for i in range(window_len, len(df) - window_len):
                if not (np.isnan(df[feats][i - window_len:i + window_len + 1].T.values.flatten()).any()):
                    x.append(
                        np.hstack((df[feats][i - window_len:i + window_len + 1].T.values.flatten(),
                                   df['label'][i],
                                   df['t'][i]))[:, None].T)
            x.append(np.zeros([window_len, (2 * window_len + 1) * len(feats) + 2]))
            x = np.vstack(x)
            data_features.append(x)

    print("***** Data successfully loaded from ", path, " for a total of ", n_larvae, " larvae samples *****")
    return data_features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to the data')
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--window', type=int)
    parser.add_argument('--idec_weights', default=None)

    args = parser.parse_args()

    create_trx(path=args.path, n_clusters=args.n_clusters, weights=args.idec_weights, window=args.window)
