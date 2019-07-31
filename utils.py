from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import backend as K
from keras import losses
import math

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
            https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
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
    # https://github.com/JHart96/keras_ordinal_categorical_crossentropy/blob/master/ordinal_categorical_crossentropy.py
    weights = K.cast(K.abs(K.argmax(y_true, axis = 1) - K.argmax(y_pred, axis = 1))/(K.int_shape(y_pred)[1] - 1), dtype = 'float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def save_probabilities(y_pred_raw, y_pred, model_name, save_name):
    pd.DataFrame(y_pred_raw).to_csv('predictions/{}_prob_{}.csv'.format(model_name, save_name), index = None)
    pd.DataFrame(y_pred).to_csv('predictions/{}_pred_{}.csv'.format(model_name, save_name), index = None, header = None)

def correntropy_loss(y_true, y_pred, sigma = 1.5):
    diff = y_true - y_pred
    out = (1 - K.exp(-1 * K.square(diff / sigma)))  # Correntropy loss
    return K.mean(out, axis = -1)

def cauchy_loss(y_true, y_pred, sigma = 1.5):
    diff = y_true - y_pred
    out = K.log(1 + K.square(diff / sigma) )  # Cauchy loss
    return K.mean(out, axis = -1)

def crop_image(im, amount_to_crop = None):
    w, h = im.size
    if amount_to_crop is None:
        amount_to_crop = abs(w - h)
    if h < w:
        l, r = math.floor(amount_to_crop/2), math.ceil(amount_to_crop/2)
        im = im.crop((l, 0, w-r, h))  # (left, upper, right, lower)
    elif w < h:
        t, b = math.floor(amount_to_crop/2), math.ceil(amount_to_crop/2)
        im = im.crop((0, t, w, h-b))
    return im

def pad_image(im, pad_ratio = 1):
    w, h = im.size
    size_diff = max(w, h) - min(w, h)
    new_size = min(w, h) + size_diff // pad_ratio  # No need to crop first
    new_im = Image.new('RGB', (new_size, new_size))  # Black by default
    t = math.floor((new_size - h)/2)
    l = math.floor((new_size - w)/2)
    new_im.paste(im, box = (l,t))  # Upper left corner
    return new_im

def preprocess_image(image_path, fill_type = 'mix', desired_size = 299):
    im = Image.open(image_path)
    try:
        if fill_type == 'crop':
            im = crop_image(im)
        elif fill_type == 'pad':
            im = pad_image(im, pad_ratio = 1)
        elif fill_type == 'mix':
            im = pad_image(im, pad_ratio = 2)
    except Exception as e:
        print("Problem opening image")
        print(e)
    im = im.resize((desired_size, )*2, resample = Image.LANCZOS)
    return im