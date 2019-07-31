import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras import losses



def save_predictions(predictions, filenames, save_name):
    filenames = [os.path.basename(x) for x in filenames]
    if predictions.dtype == int:
        category = predictions
    else:
        category = np.argmax(predictions, axis = 1)
    df = pd.DataFrame({"Id": filenames, "Expected": category})
    df.to_csv(save_name, index = None)

def save_summary(model_name, best_kappa, epoch = None, filename = 'models/performance.csv'):
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col = 0)
    else:
        df = pd.DataFrame(columns = ['Best Kappa', 'Epoch'])
    df.loc[model_name, 'Best Kappa'] = best_kappa
    if epoch is not None:
        df.loc[model_name, 'Epoch'] = epoch
    df['Epoch'] = df['Epoch'].astype(int)
    df.to_csv(filename)


def kappa_loss(y_pred, y_true, y_pow = 2, eps = 1e-10, N = 5, bsize = 16, name = 'kappa'):
        """A continuous differentiable approximation of discrete kappa loss.
            Args:
                y_pred: 2D tensor or array, [batch_size, num_classes]
                y_true: 2D tensor or array, [batch_size, num_classes]
                y_pow: int,  e.g. y_pow=2
                N: typically num_classes of the model
                bsize: batch_size of the training or validation ops
                eps: a float, prevents divide by zero
                name: Optional scope/name for op_scope.
            Returns:
                A tensor with the kappa loss."""

        with tf.name_scope(name):
            y_true = tf.to_float(y_true)
            repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
            repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
            weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
        
            pred_ = y_pred ** y_pow
            try:
                pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
            except Exception:
                pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
        
            hist_rater_a = tf.reduce_sum(pred_norm, 0)
            hist_rater_b = tf.reduce_sum(y_true, 0)
        
            conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
        
            nom = tf.reduce_sum(weights * conf_mat)
            denom = tf.reduce_sum(weights * tf.matmul(
                tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                                tf.to_float(bsize))
        
            return nom / (denom + eps)

def ordinal_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)


def save_probabilities(y_pred_raw, y_pred, model_name, save_name):
    pd.DataFrame(y_pred_raw).to_csv('predictions/{}_prob_{}.csv'.format(model_name, save_name), index = None)
    pd.DataFrame(y_pred).to_csv('predictions/{}_pred_{}.csv'.format(model_name, save_name), index = None, header = None)


def correntropy_loss(y_true, y_pred, sigma = 1.5):
    diff = y_true - y_pred
    out = (1 - K.exp(-1 * K.square(diff / sigma)))  # Correntropy loss
    return K.mean(out, axis=-1)


def cauchy_loss(y_true, y_pred, sigma = 1.5):
    diff = y_true - y_pred
    out = K.log(1 + K.square(diff / sigma) )  # Cauchy loss
    return K.mean(out, axis=-1)