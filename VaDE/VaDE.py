'''
 Old version using a dirty trick to run with keras, training is not stable and most stuff is deprecated anyway.
 Please refer to VaDE_tensorflow.py
'''

import numpy as np
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Input, Dense, Layer
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.initializers import Ones, Zeros, Constant
import scipy.io as scio
import gzip
from six.moves import cPickle
import sys

import tables

from time import time
import math
from sklearn import mixture
from sklearn.mixture import  GaussianMixture
from sklearn.cluster import KMeans
from keras.models import model_from_json
from utils import loader

import warnings
warnings.filterwarnings("ignore")

def floatX(X):
    return np.asarray(X, dtype=K.floatx())

# =====================================


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/Y_pred.size


# ==================================================


def config_init(dataset):
    if dataset == 'mnist':
        return 784, 3000, 10, 0.002, 0.002, 10, 0.9, 0.9, 1, 'sigmoid'
    if dataset == 'reuters10k':
        return 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1, 'linear'
    if dataset == 'har':
        return 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5, 'linear'
    else:
        return X.shape[-1], args.epochs, args.n_clusters, 0.002, 0.002, 10, 0.9, 0.9, 1, 'linear'


def gmmpara_init():
    
    theta_init = np.ones(n_centroid)/n_centroid
    u_init = np.zeros((latent_dim,n_centroid))
    lambda_init = np.ones((latent_dim,n_centroid))
    
    theta_p = K.variable(np.asarray(theta_init, dtype=K.floatx()), name="pi")
    u_p = K.variable(np.asarray(u_init, dtype=K.floatx()), name="u")
    lambda_p = K.variable(np.asarray(lambda_init, dtype=K.floatx()), name="lambda")
    return theta_p, u_p, lambda_p


#================================

class LatentLayer(Layer):
    """
    Gamma layer computes z thanks to reparametrization trick : z = mu + sigma
    Also implements u_p, theta_p and lambda_p as trainable weights

    # Arguments
        output_dim: number of dimensions of the latent space.
    # Input shape
        2D tensor with shape: `(n_samples, latent_dim)`.
    # Output shape
        2 * 2D tensor with shape: `(n_samples, latent_dim)`.
    """

    def __init__(self, output_dim, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LatentLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        zeros = Zeros()
        constant = Constant(value=1/n_centroid)
        ones = Ones()
        self.u_p = self.add_weight((latent_dim, n_centroid), initializer=zeros, name='u_p')
        self.theta_p = self.add_weight((n_centroid,), initializer=constant, name='theta_p')
        self.lambda_p = self.add_weight((latent_dim, n_centroid), initializer=ones, name='lambda_p')
        self.built = True

    def call(self, inputs, **kwargs):
        """ Gamma distribution of the clusters as in VaDE paper
                q(c|x) = p(c|z) = K.exp(K.sum((K.log(theta)-0.5*K.log(2*math.pi*lambda)-
                                  K.square(Z-u)/(2*lambda)),axis=1))
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q_z_x : q(z|x)
        """
        epsilon = K.random_normal(shape=(K.int_shape(inputs[1])[0] , latent_dim), mean=0.)
        q_z_x = inputs[0] + K.exp(inputs[1] / 2) * epsilon

        temp_Z = K.permute_dimensions(K.repeat(q_z_x, n_centroid), [0, 2, 1])
        temp_u = K.repeat_elements(K.expand_dims(self.u_p, 0), batch_size, axis=0)
        temp_lambda = K.repeat_elements(K.expand_dims(self.lambda_p, 0), batch_size, axis=0)
        temp_theta = K.expand_dims(K.expand_dims(self.theta_p, 0), 0) * K.ones((batch_size, latent_dim, n_centroid))

        temp_p_c_z = K.exp(K.sum((K.log(temp_theta) - 0.5 * K.log(2 * math.pi * temp_lambda) -
                                  K.square(temp_Z - temp_u) / (2 * temp_lambda)), axis=1)) + 1e-10
        self.gamma = temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

        outputs = [q_z_x, self.gamma]

        output_shapes = self.compute_output_shape([K.int_shape(i) for i in inputs])
        for o, s in zip(outputs, output_shapes):
            o.set_shape(s)
        return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return [(input_shape[0][0], latent_dim), (input_shape[0][0], n_centroid)]


# =====================================================


def vae_loss(x, x_decoded_mean):
    z_mean_t = K.permute_dimensions(K.repeat(z_mean, n_centroid),[0,2,1])
    z_log_var_t = K.permute_dimensions(K.repeat(z_log_var, n_centroid),[0,2,1])

    u_p = vade.get_layer('latent').u_p
    theta_p = vade.get_layer('latent').theta_p
    lambda_p = vade.get_layer('latent').lambda_p
    u_tensor3 = K.repeat_elements(K.expand_dims(u_p, 0), batch_size,axis=0)
    lambda_tensor3 = K.repeat_elements(K.expand_dims(lambda_p, 0), batch_size,axis=0)
    gamma = vade.get_layer('latent').gamma
    gamma_t = K.repeat(gamma, latent_dim)

    # We are trying to minimize the ELBO, i.e. maximizing its opposite
    if datatype == 'sigmoid':
        loss = alpha*original_dim * objectives.binary_crossentropy(x, x_decoded_mean)\
        + K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3 + K.square(z_mean_t-u_tensor3)/lambda_tensor3), axis=(1, 2))\
        - 0.5*K.sum(z_log_var+1,axis=-1)\
        - K.sum(K.log(K.repeat_elements(K.expand_dims(theta_p, 0),batch_size,0))*gamma,axis=-1)\
        + K.sum(K.log(gamma)*gamma,axis=-1)

    else:
        loss=alpha*original_dim * objectives.mean_squared_error(x, x_decoded_mean)\
        + K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
        - 0.5*K.sum(z_log_var+1,axis=-1)\
        - K.sum(K.log(K.repeat_elements(K.expand_dims(theta_p, 0),batch_size,0))*gamma,axis=-1)\
        + K.sum(K.log(gamma)*gamma,axis=-1)
    return loss


# ================================


def load_pretrain_weights(vade, X, Y, dataset, autoencoder=None, ae_weights=None):
    if autoencoder is None:
        ae = model_from_json(open(ae_weights).read())
        ae.load_weights('pretrain_weights/ae_'+dataset+'_weights.h5')
        vade.get_layer('encoder_0').set_weights(ae.layers[0].get_weights())
        vade.get_layer('encoder_1').set_weights(ae.layers[1].get_weights())
        vade.get_layer('encoder_2').set_weights(ae.layers[2].get_weights())
        vade.get_layer('z_mean').set_weights(ae.layers[3].get_weights())
        vade.get_layer('decoder_0').set_weights(ae.layers[-4].get_weights())
        vade.get_layer('decoder_1').set_weights(ae.layers[-3].get_weights())
        vade.get_layer('decoder_2').set_weights(ae.layers[-2].get_weights())
        vade.get_layer('output').set_weights(ae.layers[-1].get_weights())
        sample = sample_output.predict(X,batch_size=batch_size)
    else:
        autoencoder.load_weights(ae_weights)
        vade.get_layer('encoder_0').set_weights(autoencoder.layers[1].get_weights())
        vade.get_layer('encoder_1').set_weights(autoencoder.layers[2].get_weights())
        vade.get_layer('encoder_2').set_weights(autoencoder.layers[3].get_weights())
        vade.get_layer('z_mean').set_weights(autoencoder.layers[4].get_weights())
        vade.get_layer('decoder_0').set_weights(autoencoder.layers[-4].get_weights())
        vade.get_layer('decoder_1').set_weights(autoencoder.layers[-3].get_weights())
        vade.get_layer('decoder_2').set_weights(autoencoder.layers[-2].get_weights())
        vade.get_layer('output').set_weights(autoencoder.layers[-1].get_weights())
        sample = sample_output.predict(X, batch_size=batch_size)

    if dataset == 'mnist':

        gmm = GaussianMixture(n_components=n_centroid, covariance_type='diag')
        gmm.fit(sample)
        acc_0 = cluster_acc(Y, gmm.predict(sample))
        means_0 = [gmm.means_]
        for i in range(3):
            gmm.fit(sample)
            acc_0_new = cluster_acc(Y, gmm.predict(sample))
            if acc_0_new > acc_0:
                acc_0 = acc_0_new
                means_0 = gmm.means_
                covs_0 = gmm.covariances_

        K.set_value(u_p, means_0.T)
        K.set_value(lambda_p, covs_0.T)

    if dataset == 'reuters10k':
        k = KMeans(n_clusters=n_centroid)
        k.fit(sample)
        K.set_value(u_p, floatX(k.cluster_centers_.T))

    if dataset == 'har':
        g = mixture.GMM(n_components=n_centroid,covariance_type='diag',random_state=3)
        g.fit(sample)
        K.set_value(u_p, floatX(g.means_.T))
        K.set_value(lambda_p, floatX(g.covars_.T))

    if (dataset == 'custom') | (dataset is None):
        gmm = GaussianMixture(n_components=n_centroid, covariance_type='diag')
        gmm.fit(sample)
        acc_0 = cluster_acc(Y, gmm.predict(sample))
        means_0 = gmm.means_
        covs_0 = gmm.covariances_
        print(acc_0)
        print('means:', means_0.shape)
        for i in range(3):
            gmm.fit(sample)
            acc_0_new = cluster_acc(Y, gmm.predict(sample))
            if acc_0_new > acc_0:
                acc_0 = acc_0_new
                means_0 = gmm.means_
                covs_0 = gmm.covariances_

        K.set_value(u_p, means_0.T)
        K.set_value(lambda_p, covs_0.T)

    # Set trainable weights in 'latent' layer to initalized values
    K.set_value(vade.get_layer('latent').u_p, K.eval(u_p))
    K.set_value(vade.get_layer('latent').theta_p, K.eval(theta_p))
    K.set_value(vade.get_layer('latent').lambda_p, K.eval(lambda_p))

    print ('pretrain weights loaded!')
    return vade


def autoencoder(original_dim, intermediate_dim, latent_dim, act='relu', init='glorot_uniform', datatype = 'linear'):
    """
    Fully connected auto-encoder model, symmetric. Used for pretraining
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(intermediate_dim)

    # input
    x = Input(shape=(original_dim,), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks):
        h = Dense(intermediate_dim[i], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(latent_dim, kernel_initializer=init, name='encoder_%d' % (n_stacks))(
        h)  # hidden layer, features are extracted from here
    y = h

    # internal layers in decoder
    for i in range(n_stacks-1, -1, -1):
        y = Dense(intermediate_dim[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(original_dim, kernel_initializer=init, activation=datatype, name='output')(y)

    return Model(inputs=x, outputs=y, name='AE')


def pretrain_ae(vade, autoencoder, X, n_centroid, optimizer='adam', epochs=20, batch_size=256, save_dir='results/temp'):
    print('...Pretraining...')

    autoencoder.compile(optimizer=optimizer, loss='mse')

    # begin pretraining
    t0 = time()
    autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs)
    print('Pretraining time: ', time() - t0)
    autoencoder.save_weights(save_dir + '/ae_weights.h5')
    print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)

    vade.get_layer('encoder_0').set_weights(autoencoder.layers[1].get_weights())
    vade.get_layer('encoder_1').set_weights(autoencoder.layers[2].get_weights())
    vade.get_layer('encoder_2').set_weights(autoencoder.layers[3].get_weights())
    vade.get_layer('z_mean').set_weights(autoencoder.layers[4].get_weights())
    vade.get_layer('decoder_0').set_weights(autoencoder.layers[-4].get_weights())
    vade.get_layer('decoder_1').set_weights(autoencoder.layers[-3].get_weights())
    vade.get_layer('decoder_2').set_weights(autoencoder.layers[-2].get_weights())
    vade.get_layer('output').set_weights(autoencoder.layers[-1].get_weights())
    sample = sample_output.predict(X, batch_size=batch_size)

    gmm = GaussianMixture(n_components=n_centroid, covariance_type='diag')
    gmm.fit(sample)
    acc_0 = cluster_acc(Y, gmm.predict(sample))
    means_0 = gmm.means_
    covs_0 = gmm.covariances_
    print(acc_0)
    print(means_0.shape)
    for i in range(3):
        gmm.fit(sample)
        acc_0_new = cluster_acc(Y, gmm.predict(sample))
        if acc_0_new > acc_0:
            acc_0 = acc_0_new
            means_0 = gmm.means_
            covs_0 = gmm.covariances_

    K.set_value(u_p, means_0.T)
    K.set_value(lambda_p, covs_0.T)

    # Set trainable weights in 'latent' layer to initalized values
    K.set_value(vade.get_layer('latent').u_p, K.eval(u_p))
    K.set_value(vade.get_layer('latent').theta_p, K.eval(theta_p))
    K.set_value(vade.get_layer('latent').lambda_p, K.eval(lambda_p))

    return vade

# ===================================


def lr_decay():

    #### Learning rate decay
    K.set_value(adam_nn.lr, max(K.eval(adam_nn.lr)*decay_nn, 0.0002))

    print('lr_nn:%f' % K.eval(adam_nn.lr))


def epochBegin(epoch):

    if epoch % decay_n == 0 and epoch != 0:
        lr_decay()

    gamma = gamma_output.predict(X, batch_size=batch_size)
    acc = cluster_acc(np.argmax(gamma, axis=1), Y)
    global accuracy
    accuracy += [acc[0]]
    if epoch > 0:
        print ('acc_p_c_z:%0.8f' % acc[0])


class EpochBegin(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        epochBegin(epoch)

# ==============================================


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default=None, choices=['mnist', 'reuters10k', 'har', 'custom'],
                        help='Dataset')
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
                        help='Number of epochs during training')
    parser.add_argument('--epochs_pretrain', default=10, type=int,
                        help='Number of epochs during pretraining of the autoencoder')
    parser.add_argument('--latent_dim', default=10, type=int,
                        help='Dimension of the latent space')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help='Number of GPUs to use for model training')

    args = parser.parse_args()
    print(args)

    if args.dataset is not None:
        path = 'dataset/'+args.dataset+'/'
        if args.dataset == 'mnist':
            path = path + 'mnist.pkl.gz'
            if path.endswith(".gz"):
                f = gzip.open(path, 'rb')
            else:
                f = open(path, 'rb')

            if sys.version_info < (3,):
                (x_train, y_train), (x_test, y_test) = cPickle.load(f)
            else:
                (x_train, y_train), (x_test, y_test) = cPickle.load(f, encoding="bytes")

            f.close()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
            X = np.concatenate((x_train,x_test))
            Y = np.concatenate((y_train,y_test))

        if args.dataset == 'reuters10k':
            data = scio.loadmat(path+'reuters10k.mat')
            X = data['X']
            Y = data['Y'].squeeze()

        if args.dataset == 'har':
            data = scio.loadmat(path+'HAR.mat')
            X = data['X']
            X = X.astype('float32')
            Y = data['Y']-1
            X = X[:10200]
            Y = Y[:10200]
        if args.dataset == 'custom':
            if args.data is not None:
                print('Loading dataset from: ', args.data)
                try:
                    data = np.load(args.data)
                    X = data['x']
                    Y = data['y']
                    data = []
                    len_dataset = len(X)
                except:
                    dataset = tables.open_file(args.data, mode='r')
                    len_dataset = len(dataset.root.x[:])
                    print('Number of samples: ', len_dataset)
                    if len_dataset < args.update_batch:
                        X = dataset.root.x[:]
                        Y = dataset.root.y[:]
                    else:
                        X = dataset.root.x[:args.update_batch]
                        Y = dataset.root.y[:args.update_batch]
                    max_batches = int(len(X) / args.batch_size)
                    X = X[:args.batch_size * max_batches]
                    Y = Y[:args.batch_size * max_batches]

            else:
                print('You have to pass --data')

    else:
        print('Loading custom data...')
len_dataset, dataset_path = loader.load_transform(args.data, args.labels, args.lines, args.save_dir)
        print('Number of samples: ', len_dataset)
        dataset = tables.open_file(dataset_path, mode='r')
        X = dataset.root.x[:]
        Y = dataset.root.y[:]
        max_batches = int(len(X)/args.batch_size)
        X = X[:args.batch_size*max_batches]
        Y = Y[:args.batch_size * max_batches]

    batch_size = args.batch_size
    latent_dim = args.latent_dim
    intermediate_dim = [500, 500, 2000]
    K.set_floatx('float32')
    accuracy = []
    original_dim, epoch, n_centroid, lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, datatype = config_init(args.dataset)
    theta_p, u_p, lambda_p = gmmpara_init()

    # ===================
    x = Input(batch_shape=(batch_size, original_dim), name='input')
    h = Dense(intermediate_dim[0], activation='relu', name='encoder_0')(x)
    h = Dense(intermediate_dim[1], activation='relu', name='encoder_1')(h)
    h = Dense(intermediate_dim[2], activation='relu', name='encoder_2')(h)
    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)
    z = LatentLayer(output_dim=(latent_dim,), name='latent')([z_mean, z_log_var])
    h_decoded = Dense(intermediate_dim[-1], activation='relu', name='decoder_0')(z[0])
    h_decoded = Dense(intermediate_dim[-2], activation='relu', name='decoder_1')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu', name='decoder_2')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation=datatype, name='output')(h_decoded)

    # ========================
    sample_output = Model(x, z_mean)
    gamma_output = Model(x, z[1])

    # ===========================================

    vade = Model(x, x_decoded_mean)
    if args.ae_weights is not None:
        if args.dataset == 'custom':
            autoencoder = autoencoder(original_dim, intermediate_dim, latent_dim, act='relu', init='glorot_uniform',
                                      datatype=datatype)
        else:
            autoencoder = None
        vade = load_pretrain_weights(vade, X, Y, dataset, autoencoder, args.ae_weights)
    else:
        autoencoder = autoencoder(original_dim, intermediate_dim, latent_dim, act='relu', init='glorot_uniform',
                                  datatype=datatype)
        vade = pretrain_ae(vade, autoencoder, X, n_centroid, optimizer='adam', epochs=args.epochs_pretrain,
                           batch_size=batch_size, save_dir='results/temp')

    adam_nn = Adam(lr=lr_nn, epsilon=1e-4)

    vade.compile(optimizer=adam_nn, loss=vae_loss)
    vade.summary()

    epoch_begin = EpochBegin()

    # ===========================================

    vade.fit(X, X, shuffle=True, nb_epoch=epoch, batch_size=batch_size, callbacks=[epoch_begin])

    vade.save_weights('model/VADE_model_final.h5')
    gamma_output.save_weights('model/VADE_prediction_final.h5')
    sample_output.save_weights('model/VADE_encoder_final.h5')