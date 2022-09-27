import keras.backend as K
from keras.losses import binary_crossentropy
from keras import backend as K
import math
import tensorflow as tf
from sklearn.utils.extmath import cartesian
import numpy as np
import skimage

def HSmod(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    score_t = K.sum(K.abs(K.sum(y_true,2)-K.sum(y_pred,2)),1)
    A = K.cumsum(K.ones_like(y_true),2)
    score_p = K.sum(K.abs(((0.01+K.sum(y_true*A,2))/(0.01+K.sum(y_true,2)))-((0.01+K.sum(y_pred*A,2))/(0.01+K.sum(y_pred,2)))),1)

    score_tb = K.sum(K.abs(K.sum(y_true,1)-K.sum(y_pred,1)),1)
    B = K.cumsum(K.ones_like(y_true),1)
    score_pb = K.sum(K.abs(((0.01+K.sum(y_true*B,1))/(0.01+K.sum(y_true,1)))-((0.01+K.sum(y_pred*B,1))/(0.01+K.sum(y_pred,1)))),1)

    score = score_p/255 + score_t/255 + score_pb/255 + score_tb/255
    return score

def connectivity(y_true, y_pred):
    total_connectivity = 0
    for i in range(len(y_true)):
        blobs = np.max(skimage.measure.label(y_pred[i][:,:,0]))
        total_connectivity += abs(1-(blobs-1)/(blobs))
        
        #total_connectivity += abs(1-blobs)
    total_connectivity = float(total_connectivity) / len(y_true)
    return total_connectivity

def dismap_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.cast(K.clip(K.flatten(y_true), 0, 1), 'float32')
    #y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    score = (K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + smooth)
    return score

def dismap_dice(y_true, y_pred):
    smooth = 1.
    y_true = K.cast(y_true, 'float32')
    y_true_f = K.cast(K.less(K.flatten(y_true), 1.), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def dismap_c(y_true, y_pred):
    dice = dismap_TP(y_true, y_pred)
    disloss = dismap_loss(y_true, y_pred)
    return dice/2. + disloss/2.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def dice_connectivity(y_true, y_pred):
    connect = connectivity(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return connect + dice


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred):
    weight = 1.0
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred):
    smooth = 1.
    weight = 44.85
    w, m1, m2 = weight, y_true, y_pred

    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / \
        (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    weight = 10.9
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
        dice_loss(y_true, y_pred)
    return loss


resized_height = 144
resized_width = 256
max_dist = math.sqrt(resized_height**2 + resized_width**2)
n_pixels = resized_height * resized_width
all_img_locations = tf.convert_to_tensor(cartesian([np.arange(resized_height), np.arange(resized_width)]),
                                         tf.float32)
batch_size = 8

# ----------EVERYTHING BELOW THIS IS CURRENTLY NOT WORKING ----------------


def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def hausdorff_loss(y_true, y_pred):
    W = 256
    H = 192
    alpha = 1
    all_img_locations = tf.convert_to_tensor(
        cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
    max_dist = math.sqrt(W ** 2 + H ** 2)

    eps = 1e-6
    y_true = K.reshape(y_true, [W, H])
    gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
    num_gt_points = tf.shape(gt_points)[0]
    y_pred = K.flatten(y_pred)
    p = y_pred
    p_replicated = tf.squeeze(
        K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
    d_matrix = cdist(all_img_locations, gt_points)
    num_est_pts = tf.reduce_sum(p)
    term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

    d_div_p = K.min((d_matrix + eps) / (p_replicated **
                                        alpha + (eps / max_dist)), 0)
    d_div_p = K.clip(d_div_p, 0, max_dist)
    term_2 = K.mean(d_div_p, axis=0)
    print("!!!!!")
    print(term_1)
    print(term_2)
    return term_1 + term_2


def weighted_hausdorff_distance(W, H, alpha):
    all_img_locations = tf.convert_to_tensor(
        cartesian([np.arange(W), np.arange(H)]), dtype=tf.float32)
    max_dist = math.sqrt(W ** 2 + H ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [W, H])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(
                K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated **
                                                alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss
