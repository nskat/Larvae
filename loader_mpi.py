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
from mpi4py import MPI
from progress.bar import Bar

# mpiexec -n 4 python loader_v3.py D:\Nicolas\samples_screens_t15\fichiers_screns_t15\gmr_72f11_ae_01@uas_chrimson_venus_x_0070\20141218_103213

# mpiexec -np 2 singularity exec -H $HOME:/home/$USER -B /pasteur/projets/policy02/Larva-Screen/screens:/screens,/local/gensoft2/exe/openmpi tensorflow_gpu.img python /Larvae/loader_v3.py /screens/t15/GMR_72F11_AE_01@UAS_Chrimson_Venus_X_0070/r_LED100_30s2x15s30s#n#n#n@100/20140818_130752 --save_dir=/results
# mpiexec -np 2 python /Larvae/loader_v3.py /screens/t15/GMR_72F11_AE_01@UAS_Chrimson_Venus_X_0070/r_LED100_30s2x15s30s#n#n#n@100/20140818_130752 --save_dir=/results

# mpiexec -np 2 singularity exec -H $HOME:/home/$USER test.img python mpi_test.py


def load_transform(path, lines=None, save_dir='', window=1, screen=''):
    # Helper function to load the timeseries from all files in subdirectories from PATH,
    # re-scale them, compute their Fourier transforms, and return them in a numpy array,
    # along with the associated labels
    # only time samples between 30s and 90s are considered
    # labels = types of label selected : 'normal', 'large', 'strong_weak'

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

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

    feats = ['larva_length_smooth_5', 'S_smooth_5', 'eig_smooth_5', 'angle_upper_lower_smooth_5',
             'angle_downer_upper_smooth_5',  'd_eff_head_norm_smooth_5', 'd_eff_tail_norm_smooth_5',
             'As_smooth_5', 'prod_scal_1', 'prod_scal_2']

    save_path = save_dir + '/data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize hdf5 file
    Ts = 0.08
    window_len = int(np.floor(float(window) / Ts))

    # Counter
    n_larvae = 0

    if rank == 0:  # the 1st node gets the list of files to process, then gathers processed files
                    # on the fly and saves them

        save_path = save_dir + '/data'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Initialize hdf5 file
        x_shape = (0, (2 * window_len + 1) * len(feats) + 1)
        hdf5_path = save_path + '/dataset_' + lines + '.hdf5'
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        storage = hdf5_file.create_earray(hdf5_file.root, 'x', tables.Float64Atom(), shape=x_shape)
        hdf5_file.close()

    # Initialize the list of lines from the argument passed as a string
        if lines:
            lines = [x.strip() for x in lines.split(',')]
        else:
            raise FileExistsError

        # Initialize the list of lines from the argument passed as a string
        if screen == 't15':
            path = path + '/t15'
        elif screen == 't5':
            path = path + '/t5'
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

        split = np.array_split(allFiles, size - 1, axis=0)

    elif rank != 0:  # other cores
        # Create variables on other cores
        split = None

    split = comm.bcast(split, root=0)  # Broadcast split array to other cores
    comm.Barrier()

    if rank != 0:
        for file_ in split[rank - 1]:
            df = pd.read_csv(file_, sep='\t', header=None, names=names)

            # Times of the stimulus are different depending on the screen
            if screen == 't15':
                df = df[((df['t'] > 20) & (df['t'] < 50))]
            elif screen == 't5':
                df = df[((df['t'] > 40) & (df['t'] < 90))]

            # We consider 100 timestamps are a minimum
            if len(df.index) > 100:
                n_larvae += 1

                # List used to temporarily store data
                x = []

                # If necessary, removes the imaginary parts from the columns
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

                # Add 'label' column to df
                for j, label in enumerate(labels_):
                    df.loc[df[label] == 1, 'label'] = j

                df = df.reset_index(drop=True)

                # Add features to x
                for j in range(window_len, len(df) - window_len):
                    # If there are no NaNs in the window, add it
                    if not (np.isnan(df[feats][j - window_len:j + window_len + 1].T.values.flatten()).any()):
                        x.append(np.hstack((df[feats][j - window_len:j + window_len + 1].T.values.flatten(),
                                            df['label'][j]))[:, None].T)
                x = np.vstack(x)
                comm.send(x, 0, tag=1)   # send x to 0 to store in storage_tmp
        comm.send(0, 0, tag=0)

    else:
        # Loading bar
        bar = Bar('Loading files', max=len(allFiles))
        ended = 0
        for i in range(len(allFiles)):
            recv_buffer = comm.recv(source=MPI.ANY_SOURCE)
            if isinstance(recv_buffer, np.ndarray):
                storage.append(recv_buffer)
                bar.next()
            else:
                ended += 1
            if ended == size - 1:
                break
        t1 = time()

        np.random.shuffle(hdf5_file.root.x[:])

        # Shuffle data
        len_dataset = len(hdf5_file.root.x[:])
        hdf5_file.close()

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
    parser.add_argument('--save_dir')
    parser.add_argument('--lines', help='lines')
    parser.add_argument('--window')
    parser.add_argument('--screen')

    args = parser.parse_args()

    load_transform(args.path, lines=args.lines, save_dir=args.save_dir, window=args.window, screen=args.screen)