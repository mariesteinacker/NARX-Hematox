import math
import numpy as np
import tensorflow as tf
from NARX import *


class EarlyStoppAveraged(tf.keras.callbacks.Callback):
    def __init__(self, monitor, av_epochs, patience, min_delta=0.,
                 restore_best_weights=True):
        """
        Early stopping the training if specified metric increases for an
        averaged number of epochs.
        :param monitor: metric to monitor
        :param av_epochs: how many epochs to average
        :param patience: how many epochs to wait before early stopping
        :param min_delta: minimal change in metric
        :param restore_best_weights: restore the best recorded weights after
        stopping
        """
        self.monitor = monitor
        self.ae = av_epochs
        self.p = patience
        self.restore = restore_best_weights
        self.best_weights = None
        self.min_delta = min_delta

    def on_train_begin(self, logs=None):
        """
        Before training, set the best loss to infinity
        :param logs: history logs
        """
        self.best_av_loss = np.inf
        self.best_epoch = 0
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        """
        after each training epoch, check metrics and decide if to stop training
        :param epoch: current epoch
        :param logs: history logs
        """
        # only test after averaged epochs are reached
        if epoch >= self.ae:
            av_loss = tf.math.reduce_mean(
                self.model.history.history[self.monitor][-self.ae:])
            if av_loss < self.best_av_loss:
                # to not train indefinitely
                if (self.best_av_loss - av_loss < self.min_delta and
                        epoch - self.best_epoch >= self.p):
                    self.model.stop_training = True
                    if self.restore:
                        self.model.set_weights(self.best_weights)
                self.best_epoch = epoch
                self.best_av_loss = av_loss
                if self.restore:
                    self.best_weights = self.model.get_weights()
            # if no improvement has been made for a p epochs, stop training
            # as well
            if epoch - self.best_epoch >= self.p:
                self.model.stop_training = True
                if self.restore:
                    self.model.set_weights(self.best_weights)


class Prune(tf.keras.constraints.Constraint):
    def __init__(self, delta):
        """
        Magnitude pruning for a defined cut-off
        :param delta: cut-off
        """
        super(Prune, self).__init__()
        self.delta = delta

    def __call__(self, w):
        return w * tf.cast(tf.math.greater_equal(tf.math.abs(w), self.delta),
                           w.dtype)

class LP_regularizer(tf.keras.regularizers.Regularizer):
    """
    directly derived from the tf.keras implementation of L1 regularization
    A regularizer that applies an LP regularization penalty.
    The LP regularization penalty is computed as:
    loss = lp * reduce_sum((abs(x)**p)
    :param lp: Float; LP regularization factor.
    :param p: Float; regularization exponent
    """
    def __init__(self, lp=0.01, p=1., **kwargs):
        lp = kwargs.pop('l', lp)  # Backwards compatibility
        p = kwargs.pop('', p)
        if kwargs:
            raise TypeError(f'Argument(s) not recognized: {kwargs}')

        # set standard values
        lp = 0.01 if lp is None else lp
        p = 1. if p is None else p

        self.lp = tf.keras.backend.cast_to_floatx(lp)
        self.p = tf.keras.backend.cast_to_floatx(p)

    def __call__(self, x):
        x_nonzero = tf.cast(tf.boolean_mask(
            x, x != 0), tf.float32)
        return self.lp * tf.reduce_sum(
            tf.cast(tf.math.pow(tf.math.abs(x_nonzero), self.p), tf.float32))

    def get_config(self):
        return {'lp': float(self.lp), 'p': float(self.p)}


class Step_MSE_missing(tf.keras.losses.Loss):
    def __init__(self, s, name='step_mse_missing'):
        """
        Augmented MSE loss for missing data and including neighboring times
        :param s: coupling factor for neighboring times
        """
        self.s = s
        super(Step_MSE_missing, self).__init__(name=name)

    # mse with masked target
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        ymask = tf.math.is_finite(y_true)
        ymask_minus = tf.concat([ymask[1:], ymask[:1]], 0)
        ymask_plus = tf.concat([ymask[-1:], ymask[:-1]], 0)
        sol = tf.reduce_mean(
            tf.math.square(tf.cast(y_pred[ymask], tf.float32) -
                           tf.cast(y_true[ymask], tf.float32))
            + self.s * (
                    0.3 * tf.math.square(
                        tf.cast(y_pred[ymask_minus], tf.float32) - tf.cast(
                                y_true[ymask], tf.float32))
                    + 0.3 * tf.math.square(
                        tf.cast(y_pred[ymask_plus], tf.float32) - tf.cast(
                            y_true[ymask], tf.float32)))
            , axis=-1)
        return sol


class smoothing(tf.keras.losses.Loss):
    def __init__(self, k, c, name='smoothing'):
        """
        smoothing penalty function
        :param k: number of neighbors to inlcude
        :param c: coupling factor
        """
        super(smoothing, self).__init__(name=name)
        self.k = k
        self.c = c

    def call(self, y_true, y_pred):
        # k can be varied
        k = self.k
        y_dev = []
        y_pred = tf.reshape(y_pred, [-1])
        # beginning and end extra
        for i in range(k,len(y_pred)-k):
            y_dev.append(y_pred[i] - tf.math.reduce_sum(y_pred[i-k:i+k])
                         / (2*k +1))
        return tf.cast(self.c * tf.math.reduce_sum(
            tf.math.square(tf.cast(y_dev, tf.float32))), tf.float32)


class smse_smoothing(tf.keras.losses.Loss):
    def __init__(self, k, c, s, name='smse_smoothing'):
        """
        combination of smoothing penalty function and smse
        :param k: neighbors to inlcude for smoothing
        :param c: coupling factor smoothing
        :param s: coupling factor smse
        """
        super(smse_smoothing, self).__init__(name=name)
        self.k = k
        self.c = c
        self.s = s

    def call(self, y_true, y_pred):
        # smoothing part
        y_pred = tf.reshape(y_pred, [-1])
        y_dev = self.appending(y_pred)
        smooth = tf.cast(self.c * tf.math.reduce_sum(
            tf.math.square(tf.cast(y_dev, tf.float32))), tf.float32)

        # smse part
        y_true = tf.reshape(y_true, [-1])
        ymask = tf.math.is_finite(y_true)
        ymask_minus = tf.concat([ymask[1:], ymask[:1]], 0)
        ymask_plus = tf.concat([ymask[-1:], ymask[:-1]], 0)
        sol = tf.reduce_mean(
            tf.math.square(tf.cast(y_pred[ymask], tf.float32) -
                           tf.cast(y_true[ymask], tf.float32))
            + self.s * (
                    0.3 * tf.math.square(
                tf.cast(y_pred[ymask_minus], tf.float32) - tf.cast(
                    y_true[ymask], tf.float32))
                    + 0.3 * tf.math.square(
                tf.cast(y_pred[ymask_plus], tf.float32) - tf.cast(
                    y_true[ymask], tf.float32)))
            , axis=-1)

        return sol + smooth

    @tf.function
    def appending(self, y_pred):
        """
        helper function for appending smoothing penalty function, wrapped
        for speed
        """
        k = self.k
        y_dev = []
        for i in range(k, len(y_pred) - k):
            y_dev.append(y_pred[i] - tf.math.reduce_sum(y_pred[i - k:i + k]) /
                        (2 * k + 1))
        return y_dev


class Combined_loss(tf.keras.losses.Loss):
    def __init__(self, loss_list):
        """
        Combine multiple losses to one from a list
        :param loss_list: list of losses
        """
        super(Combined_loss, self).__init__()
        self.losses = loss_list
    def call(self, y_true, y_pred):
        losses = []
        for loss in self.losses:
            losses.append(loss(y_true, y_pred))
        return tf.math.reduce_sum(losses)


def buildrnn(cell, hp):
    """
    build an ARX-RNN with exponential smoothing from a specified recurrent
    cell type
    :param cell: recurrent cell type
    :param hp: hyperparameters
    :return: ARX-RNN model
    """
    # check for gaussian noise layer
    if hp['stddev'] != 0.:
        inp = tf.keras.layers.GaussianNoise(hp['stddev'], seed=0)
    else:
        inp = tf.keras.layers.Identity()

    cellb = cell(
            units=hp['units'],
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=hp['seed']), dropout=hp['in_dr'],
        kernel_constraint=Prune(hp['delta']),
            recurrent_dropout=hp['re_dr'], use_bias=hp['in_bias'],
            activation=hp['activation']
    )
    out = tf.keras.layers.Dense(units=1, activation=hp['out_act'],
        use_bias=hp['out_bias'], kernel_constraint=Prune(hp['delta']),
        kernel_regularizer=tf.keras.regularizers.L2(hp['out_l2'])
        )

    # build ARX-RNN model
    model = ARX_RNN_ES(inp=inp, cell=cellb, lags=hp['lag'],
            out=out)

    return model


def buildNARX(hp):
    """
    build an ARX-FNN from a specified MLP
    :param hp: hyperparameters
    :return: ARX-FNN model
    """
    model = ARX_net_lagged(mlp=buildmlp(hp), xlag=hp['xlag'], ylag=hp['ylag'],
                           es=hp['es'])
    return model


def buildmlp(hp):
    """
    helper function for ARX-FNN to build MLP
    :param hp: hyperparameters
    :return: MLP
    """
    if hp['stddev'] != 0.:
       inp = tf.keras.layers.GaussianNoise(hp['stddev'], seed=0)
    else:
        inp = tf.keras.layers.Identity()

    mlp = tf.keras.Sequential()
    mlp.add(inp)
    mlp.add(tf.keras.layers.Dense(units=hp['units'],
                              activation=hp['activation'],
                              name='hidden',
                              use_bias=False,
                              kernel_initializer=
                              tf.keras.initializers.RandomNormal(
                                  mean=0., stddev=.3, seed=hp['seed']),
                              kernel_regularizer=LP_regularizer(
                                  lp=hp['lp'],
                                  p=0.5
                              ),
                              kernel_constraint=Prune(hp['delta'])))
    mlp.add(
        tf.keras.layers.Dense(1, activation=hp['out_act'], name='out',
                              use_bias=hp['out_bias'],
                              kernel_initializer=
                              tf.keras.initializers.RandomNormal(
                                  mean=0., stddev=.3, seed=hp['seed']),
                              kernel_regularizer=LP_regularizer(
                                  lp=hp['lp'],
                                  p=0.5
                              ), kernel_constraint=Prune(hp['delta'])
                              ))
    return mlp