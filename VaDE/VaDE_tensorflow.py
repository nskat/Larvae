"""
Usage:
    With no transformed data:
        python VaDE_tensorflow.py --data=/path/to/data/ --n_clusters=x --epochs=200000  --save_dir=/path/to/save_dir --epochs_pretraining=100
    With data already transformed and stored in a hdf5 file:
        python VaDE_tensorflow.py --data=/path/to/data.hdf5 --n_clusters=x --epochs=200000  --save_dir=/path/to/save_dir --epochs_pretraining=100
    If you also have pretrained weights for the autoencoder:
        python VaDE_tensorflow.py custom  --data=/path/to/data.hdf5 --n_clusters=x --epochs=200000  --save_dir=/path/to/save_dir --ae_weights=path/to/pretrained/ae.h5
"""

# Ugly hack to allow absolute import from the root folder
# whatever its name is. Please forgive the heresy.
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import objectives
from sklearn import metrics
import tables

from time import time
import math
from sklearn.mixture import GaussianMixture
from utils import loader

import warnings
warnings.filterwarnings("ignore")


#python VaDE_tensorflow.py --data=D:\Nicolas\samples_screens_t15\fichiers_screns_t15\gmr_72f11_ae_01@uas_chrimson_venus_x_0070\20141218_103213 --epochs=5000 --epochs_pretrain=50

# =====================================

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size
             
# ==================================================

def autoencoder(dims, act='relu', init='glorot_uniform', datatype = 'linear'):
    """
    Fully connected auto-encoder model, symmetric. Used for pretraining
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1

    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks - 1):
        h = Dense(dims[i+1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks-1))(h)  # hidden layer, features are extracted from here
    y = h

    # internal layers in decoder
    for i in range(n_stacks-1):
        y = Dense(dims[n_stacks - 1 - i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, activation=act, name='output')(y)

    return Model(inputs=x, outputs=y, name='AE')


# Model class
class vaDE(object):
    def __init__(self, dataset, dimensions, n_clusters, batch_size=256, epochs_pretraining=20, weights=None,
                 save_dir='/results/tmp', learning_rate=0.002, alpha=1, dropout=1.0, datatype='numeric',
                 initialize=True):
        self.dataset = dataset
        if dataset:
            self.len_dataset = len(self.dataset.root.x[:])
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.datatype = datatype
        self.alpha = alpha
        self.dropout = dropout
        self.theta_p = tf.get_variable("pi", self.n_clusters,
                                       initializer=tf.constant_initializer(1.0 / self.n_clusters))
        self.theta_p_norm = tf.get_variable("pi_norm", self.n_clusters,
                                            initializer=tf.constant_initializer(1.0 / self.n_clusters))
        self.u_p = tf.get_variable("u", (self.dimensions[-1], self.n_clusters),
                                   initializer=tf.constant_initializer(0.0))
        self.lambda_p = tf.get_variable("lambda", (self.dimensions[-1], self.n_clusters),
                                        initializer=tf.constant_initializer(1.0))

        # Input
        self.x = tf.placeholder(tf.float32, shape=[None, dimensions[0]])

        # Create and pretrain autoencoder (or load weights if available)
        if initialize:
            self.pretrain_ae(weights, optimizer='adam', epochs=epochs_pretraining, save_dir=save_dir)
        else:
            self.autoencoder = autoencoder(self.dimensions, act='relu', init='glorot_uniform', datatype=self.datatype)

        # Create network
        self.create_network()

        # Create loss and optimizer
        self.create_loss_optimizer()

        # Variables initializer
        init = tf.global_variables_initializer()

        # Launch session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        # Initialize clustering variables
        if initialize:
            self.initialize_variables()

        self.saver = tf.train.Saver()


    def create_network(self):
        network_weights = self.initialize_weights()

        # Create encoder network

        self.l1 = tf.nn.relu(self.fc_layer(self.x, network_weights['encoder_0']['W'], network_weights['encoder_0']['b'],
                                           keep_prob=self.dropout))
        self.l2 = tf.nn.relu(self.fc_layer(self.l1, network_weights['encoder_1']['W'], network_weights['encoder_1']['b'],
                                           keep_prob=self.dropout))
        self.l3 = tf.nn.relu(self.fc_layer(self.l2, network_weights['encoder_2']['W'], network_weights['encoder_2']['b'],
                                           keep_prob=self.dropout))

        # Create latent space layers
        self.z_mean = self.fc_layer(self.l3, network_weights['z_mean']['W'], network_weights['z_mean']['b'])
        self.z_log_var = self.fc_layer(self.l3, network_weights['z_log_var']['W'], network_weights['z_log_var']['b'])
        self.q_z_x = self.latent_layer(self.z_mean, self.z_log_var)
        self.gamma = self.get_gamma(self.q_z_x)
        self.gamma_norm = self.get_gamma_norm(self.q_z_x)

        # create decoder network
        self.l5 = tf.nn.relu(self.fc_layer(self.q_z_x, network_weights['decoder_0']['W'], network_weights['decoder_0']['b'],
                                           keep_prob=self.dropout))
        self.l6 = tf.nn.relu(self.fc_layer(self.l5, network_weights['decoder_1']['W'], network_weights['decoder_1']['b'],
                                           keep_prob=self.dropout))
        self.l7 = tf.nn.relu(self.fc_layer(self.l6, network_weights['decoder_2']['W'], network_weights['decoder_2']['b'],
                                           keep_prob=self.dropout))
        self.output = tf.nn.relu(self.fc_layer(self.l7, network_weights['output']['W'], network_weights['output']['b'],
                                               keep_prob=self.dropout))
        print('Network created.')

    def create_loss_optimizer(self):
        z_mean_t = K.permute_dimensions(K.repeat(self.z_mean, self.n_clusters), [0, 2, 1])
        z_log_var_t = K.permute_dimensions(K.repeat(self.z_log_var, self.n_clusters), [0, 2, 1])

        u_tensor3 = K.repeat_elements(K.expand_dims(self.u_p, 0), self.batch_size, axis=0)
        lambda_tensor3 = K.repeat_elements(K.expand_dims(self.lambda_p, 0), self.batch_size, axis=0)
        gamma_t = K.repeat(self.gamma, self.dimensions[-1])

        if self.datatype == 'binary':
            self.loss = self.alpha * self.dimensions[0] * objectives.binary_crossentropy(self.x, self.output) \
                   + K.sum(0.5 * gamma_t * (self.dimensions[-1] * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                   - 0.5 * K.sum(self.z_log_var + 1, axis=-1) \
                   - K.sum(K.log(K.repeat_elements(K.expand_dims(self.theta_p, 0), self.dimensions[0], 0)) * self.gamma, axis=-1) \
                   + K.sum(K.log(self.gamma) * self.gamma, axis=-1)

        else:
            self.loss = self.alpha * self.dimensions[0] * objectives.mean_squared_error(self.x, self.output) \
                   + K.sum(0.5 * gamma_t * (self.dimensions[-1] * K.log(math.pi * 2) + K.log(lambda_tensor3) + K.exp(z_log_var_t) / lambda_tensor3 + K.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2)) \
                   - 0.5 * K.sum(self.z_log_var + 1, axis=-1) \
                   - K.sum(K.log(K.repeat_elements(K.expand_dims(self.theta_p_norm, 0), self.batch_size, 0)) * self.gamma, axis=-1) \
                   + K.sum(K.log(self.gamma) * self.gamma, axis=-1)
        self.cost = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9, beta2=0.999, epsilon=5).minimize(self.cost)

        self.normalize = tf.assign(self.theta_p_norm, self.theta_p/tf.reduce_sum(self.theta_p))

        print('Loss optimizer created.')


    def initialize_weights(self):
        weights = dict()

        n_stacks = len(self.dimensions) - 1

        # Encoding layers weights
        for i in range(n_stacks - 1):
            weights['encoder_%d' % i] = {
                        'W': tf.Variable(initial_value=self.autoencoder.get_layer('encoder_%d' % i).get_weights()[0],
                                         name='W%d' % i),
                        'b': tf.Variable(initial_value=self.autoencoder.get_layer('encoder_%d' % i).get_weights()[1],
                                         name='b%d' % i)}
        # Latent space layers weights
        weights['z_mean'] = {
            'W': tf.Variable(initial_value=self.autoencoder.get_layer('encoder_3').get_weights()[0],
                             name='Wz_mean'),
            'b': tf.Variable(initial_value=self.autoencoder.get_layer('encoder_3').get_weights()[1],
                             name='bz_mean')}
        weights['z_log_var'] = {
            'W': tf.get_variable(name='Wz_log_var', shape=[self.dimensions[3], self.dimensions[4]],
                                 initializer=tf.contrib.layers.xavier_initializer()),
            'b': tf.get_variable(name='bz_log_var', shape=self.dimensions[4],
                                 initializer=tf.constant_initializer(0.0))}

        # Decoding layers weights
        for i in range(n_stacks - 1):
            weights['decoder_%d' % i] = {
                'W': tf.Variable(initial_value=self.autoencoder.get_layer('decoder_%d' % i).get_weights()[0],
                                 name='W%d' % (i+n_stacks)),
                'b': tf.Variable(initial_value=self.autoencoder.get_layer('decoder_%d' % i).get_weights()[1],
                                 name='b%d' % (i+n_stacks))}

        weights['output'] = {
            'W': tf.Variable(initial_value=self.autoencoder.get_layer('output').get_weights()[0],
                             name='W7'),
            'b': tf.Variable(initial_value=self.autoencoder.get_layer('output').get_weights()[1],
                             name='b7')}
        print('Weights loaded from pretrained autoencoder.')
        return weights

    def initialize_variables(self):
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
        if self.len_dataset < 2000000:
            sample = self.transform(self.dataset.root.x[:, :-1])
            Y = self.dataset.root.x[:, -1].astype('int')
        else:
            sample = self.transform(self.dataset.root.x[:2000000, :-1])
            Y = self.dataset.root.x[:2000000, -1].astype('int')

        gmm.fit(sample)
        acc_0 = cluster_acc(Y, gmm.predict(sample))
        means_0 = gmm.means_
        covs_0 = gmm.covariances_
        for i in range(3):
            gmm.fit(sample)
            acc_0_new = cluster_acc(Y, gmm.predict(sample))
            if acc_0_new > acc_0:
                acc_0 = acc_0_new
                means_0 = gmm.means_
                covs_0 = gmm.covariances_
        self.sess.run(self.u_p.assign(means_0.T))
        self.sess.run(self.lambda_p.assign(covs_0.T))
        print('Accuracy at beginning:', acc_0)

    def fc_layer(self, prev, W, b, keep_prob=1.0):
        return tf.nn.dropout(tf.matmul(prev, W) + b, keep_prob=keep_prob)

    def latent_layer(self, z_mean, z_log_var):
        epsilon = tf.random_normal(shape=tf.shape(z_mean), mean=0.)
        return z_mean + tf.exp(z_log_var / 2) * epsilon

    def get_gamma(self, q_z_x):
        temp_Z = tf.keras.backend.permute_dimensions(tf.keras.backend.repeat(q_z_x, self.n_clusters), [0, 2, 1])
        temp_u = tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(self.u_p, 0),
                                                  self.batch_size, axis=0)
        temp_lambda = tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(self.lambda_p, 0),
                                                       self.batch_size, axis=0)
        temp_theta = tf.keras.backend.expand_dims(tf.keras.backend.expand_dims(self.theta_p, 0), 0) * \
                     tf.keras.backend.ones((self.batch_size, self.dimensions[-1], self.n_clusters))
        temp_p_c_z = K.exp(K.sum((K.log(temp_theta) - 0.5 * K.log(2 * math.pi * temp_lambda) -
                                  K.square(temp_Z - temp_u) / (2 * temp_lambda)), axis=1)) + 1e-10
        return temp_p_c_z / tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)

    def get_gamma_norm(self, q_z_x):
        temp_Z = tf.keras.backend.permute_dimensions(tf.keras.backend.repeat(q_z_x, self.n_clusters), [0, 2, 1])
        temp_u = tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(self.u_p, 0),
                                                  self.batch_size, axis=0)
        temp_lambda = tf.keras.backend.repeat_elements(tf.keras.backend.expand_dims(self.lambda_p, 0),
                                                       self.batch_size, axis=0)

        temp_theta = tf.keras.backend.expand_dims(tf.keras.backend.expand_dims(self.theta_p_norm, 0), 0) * \
                     tf.keras.backend.ones((self.batch_size, self.dimensions[-1], self.n_clusters))
        temp_p_c_z = K.exp(K.sum((K.log(temp_theta) - 0.5 * K.log(2 * math.pi * temp_lambda) -
                                  K.square(temp_Z - temp_u) / (2 * temp_lambda)), axis=1)) + 1e-10
        return temp_p_c_z / tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)

    def train_on_batch(self, X):
        opt, cost, _ = self.sess.run((self.optimizer, self.cost, self.normalize), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        res = np.empty([0, self.dimensions[-1]])
        batches_per_epochs = int(len(X) / self.batch_size)
        for i in range(batches_per_epochs):
            batch = loader.generate_batch(X, self.len_dataset, i, self.batch_size)
            res = np.vstack((res, self.sess.run(self.z_mean, feed_dict={self.x: batch})))
        return res

    def predict(self, X):
        # Predict the classes for elements of X; the computation is done in batches
        res = np.array([])
        res_norm = np.array([])
        batches_per_epochs = int(np.floor(len(X) / self.batch_size))
        for i in range(batches_per_epochs):
            batch = X[self.batch_size * i:self.batch_size * (i + 1)]
            res = np.append(res, np.argmax(self.sess.run(self.gamma, feed_dict={self.x: batch}), axis=1))
            res_norm = np.append(res_norm,
                                 np.argmax(self.sess.run(self.gamma_norm, feed_dict={self.x: batch}), axis=1))
        batch = X[self.batch_size * batches_per_epochs:]
        remaining = self.batch_size - len(batch)
        batch = np.vstack((batch, np.zeros([remaining, batch.shape[-1]])))
        res = np.append(res, np.argmax(self.sess.run(self.gamma, feed_dict={self.x: batch}), axis=1))
        return res.astype('int'), res_norm.astype('int')

    def pretrain_ae(self, weights, optimizer='adam', epochs=20, save_dir='/results/tmp'):

        self.autoencoder = autoencoder(self.dimensions, act='relu', init='glorot_uniform', datatype=self.datatype)

        if weights is not None:
            self.autoencoder.load_weights(weights)
            return

        print('Pretraining...')

        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        # begin pretraining
        t0 = time()
        if self.len_dataset < 2000000:
            X = self.dataset.root.x[:, :-1]
            self.autoencoder.fit(X, X, batch_size=self.batch_size, epochs=epochs)
        else:
            steps = int(self.len_dataset/self.batch_size)
            self.autoencoder.fit_generator(loader.generate_data_ae(self.dataset, batch_size, self.len_dataset),
                                           steps_per_epoch=steps, epochs=epochs, verbose=1,
                                           shuffle=True, initial_epoch=0)
        print('Pretraining time: ', time() - t0)
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)

    def save_model(self, save_dir, i):
        save_path = save_dir + "/tmp/model_%d" % i
        input = {'x': self.x}
        outputs = {'z_mean': self.z_mean,
                   'gamma': self.gamma}

        saved = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s" % saved)

    def load_model(self, save_dir):
        self.saver.restore(self.sess, save_dir)


# ==============================================


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default=None,
                        help='Path to dataset')
    parser.add_argument('--n_clusters', default=6, type=int,
                        help='Number of clusters to find')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size during both autoencoder pretraining an training')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Coefficient of the clustering loss')
    parser.add_argument('--dropout', default=1.0, type=float,
                        help='Percentage of weights to keep during dropout')
    parser.add_argument('--update_batch', default=2000000, type=int,
                        help='Quantity of data that can fit into memory, typically several million rows')
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='/results')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs during training')
    parser.add_argument('--epochs_pretraining', default=10, type=int,
                        help='Number of epochs during pretraining of the autoencoder')
    parser.add_argument('--latent_dim', default=10, type=int,
                        help='Dimension of the latent space')

    args = parser.parse_args()
    print(args)

    if args.dataset is not None:
        dataset = tables.open_file(args.dataset, mode='r')
        len_dataset = len(dataset.root.x[:])
        print('Number of samples: ', len_dataset)
        if len_dataset < args.update_batch:
            X = dataset.root.x[:, :-1]
            Y = dataset.root.x[:, -1].astype('int')
        else:
            X = dataset.root.x[:args.update_batch, :-1]
            Y = dataset.root.x[:args.update_batch, -1].astype('int')

    else:
        print('You have to pass a --dataset')
        raise FileNotFoundError

    batch_size = args.batch_size
    batches_per_epochs = int(len_dataset / args.batch_size)

    dimensions = [X.shape[-1], 500, 500, 2000, args.latent_dim]

    vade = vaDE(dataset=dataset, dimensions=dimensions, n_clusters=args.n_clusters, batch_size=args.batch_size,
                epochs_pretraining=args.epochs_pretraining, weights=args.ae_weights,
                save_dir=args.save_dir, learning_rate=0.02, alpha=1, dropout=args.dropout, datatype='numeric')

    prec, prec_norm = vade.predict(X)

    for i in range(args.epochs * batches_per_epochs):
        batch = loader.generate_batch(dataset, len_dataset, i, batch_size)
        vade.train_on_batch(batch)

        if (i % 10000 * batches_per_epochs) == 0:
            prediction, prediction_norm = vade.predict(X)
            acc = cluster_acc(prediction_norm, Y)
            nmi = np.round(metrics.normalized_mutual_info_score(Y, prediction_norm), 5)
            ari = np.round(metrics.adjusted_rand_score(Y, prediction_norm), 5)

            print('Iter', i, ': Acc', acc, ', nmi', nmi, ', ari', ari)
            vade.save_model(args.save_dir, i)
            if np.isnan(vade.sess.run(vade.lambda_p).sum().sum()):
                break
        # Stop the training if the points stop moving
            if np.sum(prec != prediction_norm) < (args.tol * len(X)):
                vade.save_model(args.save_dir, i)
                print('Tolerance threshold reached, stopping training.')
                break
            prec = prediction_norm
