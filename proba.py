import numpy as np
import glob
import os
import pandas as pd
from DEC import autoencoder
from IDEC_GPU import ClusteringLayer
from keras.models import Model
from progress.bar import Bar


#python proba.py --path=D:/Nicolas/samples_screens_t15/fichiers_screns_t15/gmr_72f11_ae_01@uas_chrimson_venus_x_0070 --idec_weights=D:\Nicolas\Python_projects\results_cluster\tests\window_input_2s_pos_GMR_72F11_6_clusters\IDEC_model_final.h5 --n_clusters=6 --window=1 --screen=t15
#python proba.py --path=D:\Nicolas\Python_projects\results_cluster\GMR_75G10_AE_01@UAS_TNT_2_0003 --idec_weights=D:\Nicolas\Python_projects\results_cluster\tests\window_input_2s_pos_GMR_75G10_FCF_6_clusters\IDEC_model_final.h5 --n_clusters=6 --window=1 --screen=t5

def load_probas(path='', idec_weights='', n_clusters=6, window=1, tau=5, screen='t15', tag=''):
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

    len_autocorr = int(float(tau)/Ts)
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
    autocorrelations = []

    for dirs, _, _ in os.walk(path):
        files += glob.glob(dirs + r"/State_Amplitude_t*.txt")

    if files:
        bar = Bar('Computing probabilities...', max=len(files))

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
                        # bar.next()
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

                assert len(x) <= len(proba)
                assert len(x) == len(time)

                res = modele.predict(x)
                predictions = res[0].argmax(axis=1)

                autocorr = [np.corrcoef(predictions, predictions)[0, 1]]
                for i in range(1, len(predictions) - 1):
                    tmp = np.corrcoef(predictions[:-i], predictions[i:])[0, 1]
                    if not np.isnan(tmp):
                        autocorr.append(tmp)
                if len(autocorr) > len_autocorr:
                    autocorrelations.append(np.array(autocorr[:len_autocorr]))

                for i in range(len(x)):
                    time_index = min(int((time[i] - start_time) / Ts) - 1, len(proba) - 1)
                    proba[time_index, predictions[i]] += 1
            bar.next()
        print('//////////////////////')
        print(np.sum(proba, axis=0))
        proba = proba / np.sum(proba, axis=1)[:, None]
        print('//////////////////////')
        print(np.sum(proba, axis=0))
        times = np.arange(start_time, end_time, Ts)[:len(proba)]
        autocorrelations = np.array(autocorrelations)
        autocorrelations = np.mean(autocorrelations, axis=0)

        save_dir = path + "/probas_" + tag
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.savez_compressed(save_dir + '/probas_' + str(n_clusters) + '_clusters.npz',
                            x=proba, t=times, ac=autocorrelations)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='compute probabilities',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='Path to data')
    parser.add_argument('--idec_weights', help='Path to model weights')
    parser.add_argument('--n_clusters', help='Number of clusters')
    parser.add_argument('--window')
    parser.add_argument('--tau', help='Length of autocorrelation')
    parser.add_argument('--screen')
    parser.add_argument('--tag')

    args = parser.parse_args()

    load_probas(path=args.path, idec_weights=args.idec_weights, n_clusters=args.n_clusters,
                window=args.window, tau=args.tau, screen=args.screen, tag=args.tag)
