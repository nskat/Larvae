"""
Based on the implementation for Improved Deep Embedded Clustering by Xifeng Guo, as described in paper:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure
        Preservation. IJCAI 2017.

Usage:
    With no transformed data:
        python IDEC_GPU.py None --data=path/to/data/directories
    With data already transformed and stored in a hdf5 or npz file:
        python IDEC_GPU.py custom --data=path/to/data.hdf5
    If you also have pretrained weights for the autoencoder:
        python IDEC_GPU.py custom --data=path/to/data.hdf5 --ae_weights=path/to/pretrained/ae.h5

"""

from time import time
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
import csv
import os

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import tables
from DEC import cluster_acc, ClusteringLayer, autoencoder
import loader_v2

# python /mnt/Larvae/IDEC_GPU.py --data=/pasteur/projets/policy02/Larva-Screen/screens/t15 --lines=gmr_72f11_ae_01@uas_chrimson_venus_x_0070\20141218_103213 --save_dir=/pasteur/homes/nskatchk/test --n_gpus=8

# python Larvae/IDEC_GPU.py --dataset=custom --data=/home/dgxuser/Nicolas/data/dataset.hdf5 --ae_weights=/home/dgxuser/Nicolas/results_pretraining/ae_weights.h5 --save_dir=/home/dgxuser/Nicolas

class IDEC(object):
    def __init__(self,
                 x,
                 y,
                 dims,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=256,
                 n_gpus=1,
                 init_clustering='kmeans',
                 update_batch=2000000,
                 epochs=5,
                 init='glorot_uniform',
                 save_dir='/results'):

        super(IDEC, self).__init__()
        self.x = x
        self.y = y
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.batch_size = batch_size*n_gpus
        self.n_gpus = n_gpus
        self.init_clustering = init_clustering
        self.epochs = epochs
        self.update_batch = update_batch
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        if self.n_gpus > 1:
            self.multi_gpu_autoencoder = multi_gpu_model(self.autoencoder, gpus=self.n_gpus, cpu_merge=True, cpu_relocation=False)
        self.save_dir = save_dir
        self.pretrained = False

    def initialize_model(self, ae_weights=None, optimizer='adam'):
        if ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
        else:
            self.pretrain(optimizer=optimizer)
            print('Autoencoder pretrained successfully')

        hidden = self.autoencoder.get_layer(name='encoder_%d' % (self.n_stacks - 1)).output
        self.encoder = Model(inputs=self.autoencoder.get_layer(name='input').input, outputs=hidden)

        # prepare IDEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

        self.model = Model(inputs=self.autoencoder.get_layer(name='input').input,
                           outputs=[clustering_layer, self.autoencoder.get_layer('decoder_0').output])
        # If the model is single GPU compile now, otherwise it will be done later
        self.model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                           loss_weights=[args.gamma, 1],
                           optimizer=optimizer)

    def pretrain(self, optimizer='adam'):
        save_dir_pretraining = self.save_dir + '/results_pretraining'
        if not os.path.exists(save_dir_pretraining):
            os.makedirs(save_dir_pretraining)
        print('...Pretraining...')
        t1 = time()

        if self.n_gpus == 1:
            self.autoencoder.compile(optimizer=optimizer, loss='mse')
            self.autoencoder.fit(self.x, self.x, batch_size=self.batch_size, epochs=self.epochs)

        else:
            self.multi_gpu_autoencoder.compile(optimizer=self.optimizer, loss='mse')
            self.multi_gpu_autoencoder.fit(self.x, self.x, batch_size=self.batch_size, epochs=self.epochs)

        print('Pretraining time: ', time() - t1)
        self.autoencoder.save_weights(save_dir_pretraining + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir_pretraining)
        self.pretrained = True

    def load_weights(self, weights_path):  # load weights of IDEC model
        self.model.load_weights(weights_path)

    @staticmethod
    def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def clustering(self,
                   dataset,
                   tol=1e-3,
                   update_interval=140,
                   maxiter=2e4,
                   optimizer='adam'):

        save_dir_clustering = self.save_dir + '/results_clustering'
        if not os.path.exists(save_dir_clustering):
            os.makedirs(save_dir_clustering)
        print('Update interval', update_interval)
        save_interval = int(self.x.shape[0] / self.batch_size * 2)  # 2 epochs
        print('Save interval', save_interval)

        # initialize cluster centers using k-means
        if self.init_clustering == 'kmeans':
            print('Initializing cluster centers with k-means.')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(self.encoder.predict(self.x))
            y_pred_last = y_pred
            self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        elif self.init_clustering == 'GMM':
            print('Initializing cluster centers with GMM.')
            gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
            sample = self.encoder.predict(self.x)
            gmm.fit(sample)
            acc_0 = cluster_acc(self.y, gmm.predict(sample))
            weights_0 = [gmm.means_]
            for i in range(3):
                gmm.fit(sample)
                acc_0_new = cluster_acc(self.y, gmm.predict(sample))
                if acc_0_new > acc_0:
                    acc_0 = acc_0_new
                    weights_0 = [gmm.means_]
            y_pred = gmm.predict(sample)
            y_pred_last = y_pred
            self.model.get_layer(name='clustering').set_weights(weights_0)

        # Compile model if it has multiple GPUS
        if self.n_gpus != 1:
            self.multi_model = multi_gpu_model(self.model, gpus=self.n_gpus)
            self.multi_model.compile(loss={'clustering': 'kld', 'decoder_0': 'mse'},
                                     loss_weights=[args.gamma, 1],
                                     optimizer=optimizer)

        # logging file
        if not os.path.exists(save_dir_clustering):
            os.makedirs(save_dir_clustering)
        with open(save_dir_clustering + '/idec_log.csv', 'w') as logfile:
            logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
            logwriter.writeheader()

            loss = [0, 0, 0]
            index = 0
            index_update = 0

            for ite in range(int(maxiter)):
                # If the dataset cannot fit in memory, do the training on batches successively
                if len_dataset > self.update_batch:
                    # At each epoch, train on the next batch
                    if index == 0:
                        if (index_update + 1) * self.update_batch > len_dataset:
                            self.x = dataset.root.x[index_update * self.update_batch:, :-1]
                            self.y = dataset.root.x[index_update * self.update_batch:, -1]
                            index_update = 0

                        else:
                            self.x = dataset.root.x[index_update * self.update_batch: (index_update + 1) * self.update_batch, :-1]
                            self.y = dataset.root.x[index_update * self.update_batch: (index_update + 1) * self.update_batch, -1]
                            index_update += 1

                if ite % update_interval == 0:

                    if self.n_gpus == 1:
                        q, _ = self.model.predict(self.x, verbose=0)
                        p = self.target_distribution(q)  # update the auxiliary target distribution p
                    else:
                        q, _ = self.multi_model.predict(self.x, verbose=0)
                        p = self.target_distribution(q)  # update the auxiliary target distribution p

                    # evaluate the clustering performance
                    y_pred = q.argmax(1)
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

                    y_pred_last = y_pred
                    if y is not None:
                        acc = np.round(cluster_acc(self.y, y_pred), 5)
                        nmi = np.round(metrics.normalized_mutual_info_score(self.y, y_pred), 5)
                        ari = np.round(metrics.adjusted_rand_score(self.y, y_pred), 5)
                        loss = np.round(loss, 5)
                        logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                        logwriter.writerow(logdict)
                        print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss)

                    # # check stop criterion
                    if ite > 0 and delta_label == 0:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print('Reached tolerance threshold. Stopping training.')
                        logfile.close()
                        break

                # train on batch
                    if self.n_gpus == 1:
                        if (index + 1) * self.batch_size > x.shape[0]:

                            loss = self.model.train_on_batch(x=self.x[index * self.batch_size::],
                                                             y=[p[index * self.batch_size::], self.x[index * self.batch_size::]])
                            index = 0
                        else:
                            loss = self.model.train_on_batch(x=self.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                             y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                                self.x[index * self.batch_size:(index + 1) * self.batch_size]])
                    else:
                        if (index + 1) * self.batch_size > x.shape[0]:

                            loss = self.multi_model.train_on_batch(x=self.x[index * self.batch_size::],
                                                                   y=[p[index * self.batch_size::], self.x[index * self.batch_size::]])
                            index = 0
                        else:
                            loss = self.multi_model.train_on_batch(x=self.x[index * self.batch_size:(index + 1) * self.batch_size],
                                                                   y=[p[index * self.batch_size:(index + 1) * self.batch_size],
                                                                   self.x[index * self.batch_size:(index + 1) * self.batch_size]])
                    index += 1

                # save intermediate model
                if ite % save_interval == 0:
                    # save IDEC model checkpoints
                    print('saving model to:', save_dir_clustering + '/IDEC_model_' + str(ite) + '.h5')
                    self.model.save_weights(save_dir_clustering + '/IDEC_model_' + str(ite) + '.h5')

                ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir_clustering + '/IDEC_model_final.h5')
        self.model.save_weights(save_dir_clustering + '/IDEC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default=None, choices=['custom', None],
                        help='Dataset, if None you must pass a path to --data')
    parser.add_argument('--data', default=None,
                        help='Path to the data, if None you must pass a --dataset')
    parser.add_argument('--lines', default=None,
                        help='Lines to train on, usage: --lines=line1,line2,linex; ' +
                             'if None, will take all of those in --data')
    parser.add_argument('--n_clusters', default=6, type=int,
                        help='Number of clusters to find')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size during both autoencoder pretraining an training')
    parser.add_argument('--maxiter', default=50000, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Coefficient of the clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--update_batch', default=2000000, type=int,
                        help='Quantity of data that can fit into memory, typically several million rows')
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--labels', default='normal', choices=['normal', 'large', 'strong_weak'],
                        help='Types of label selected')
    parser.add_argument('--save_dir', default='/results')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs during pretraining of the autoencoder')
    parser.add_argument('--latent_dim', default=10, type=int,
                        help='Dimension of the latent space')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help='Number of GPUs to use for model training')
    parser.add_argument('--init', default='GMM', choices=['kmeans', 'GMM'],
                        help='Initialization of clusters centers')

    args = parser.parse_args()
    print(args)

    # Create dir to store results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    if not args.dataset:
        print('Transforming and loading data from: ' + args.data)
        if args.data is not None:
            len_dataset, dataset_path = loader_v2.load_transform(args.data, args.labels, args.lines, args.save_dir)
            print('Number of samples: ', len_dataset)
            dataset = tables.open_file(dataset_path, mode='r')
            if len_dataset < args.update_batch:
                x = dataset.root.x[:, :-1]
                y = dataset.root.x[:, -1]
            else:
                x = dataset.root.x[:args.update_batch, :-1]
                y = dataset.root.x[:args.update_batch, -1]

        else:
            print('You have to pass --data')
            exit()

    elif args.dataset == 'custom':
        if args.data is not None:
            print('Loading dataset from: ', args.data)
            try:
                data = np.load(args.data)
                x = data['x']
                y = data['y']
                data = []
                len_dataset = len(x)
            except:
                dataset = tables.open_file(args.data, mode='r')
                len_dataset = len(dataset.root.x[:])
                print('Number of samples: ', len_dataset)
                if len_dataset < args.update_batch:
                    x = dataset.root.x[:, :-1]
                    y = dataset.root.x[:, -1]
                else:
                    x = dataset.root.x[:args.update_batch, :-1]
                    y = dataset.root.x[:args.update_batch, -1]
        else:
            print('You have to pass --data')

    # prepare the IDEC model
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.5, amsgrad=False)
    idec = IDEC(x, y, dims=[x.shape[-1], 500, 500, 2000, args.latent_dim], n_clusters=args.n_clusters,
                batch_size=args.batch_size, n_gpus=args.n_gpus,
                init_clustering=args.init, update_batch=args.update_batch, epochs=args.epochs, save_dir=args.save_dir)
    idec.initialize_model(ae_weights=args.ae_weights, optimizer=optimizer)
    idec.model.summary()

    # begin clustering, time not include pretraining part.
    t0 = time()
    y_pred = idec.clustering(dataset, tol=args.tol, maxiter=args.maxiter,
                             update_interval=args.update_interval, optimizer=optimizer)
