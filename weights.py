from keras.utils.data_utils import get_file
import keras
from keras.models import *
from keras.layers import *


WEIGHTS_DIR = "/home/zijue/Downloads/"

resnet34 = {
    'model': 'resnet34',
    'dataset': 'imagenet',
    'classes': 1000,
    'include_top': False,
    'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
    'name': 'resnet34_imagenet_1000.h5',
    'md5': '2ac8277412f65e5d047f255bcbd10383',
}

resnet34_notop = {
    'model': 'resnet34_notop',
    'dataset': 'imagenet',
    'classes': 1000,
    'include_top': False,
    'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
    'name': 'resnet34_imagenet_1000_no_top.h5',
    'md5': '8caaa0ad39d927cb8ba5385bf945d582',
}

vgg16 = {'name': 'vgg16_weights_th_dim_ordering_th_kernels.h5',
         'url': "https://github.com/fchollet/deep-learning-models/"
         "releases/download/v0.1/"
         "vgg16_weights_th_dim_ordering_th_kernels.h5"}

vgg16_notop = {'name': 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
               'url': "https://github.com/fchollet/deep-learning-models/"
               "releases/download/v0.1/"
               "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"}


def pretrain_resnet34(model, top=True, skip=True):
    weights = None
    if top == True:
        try:
            weights = WEIGHTS_DIR+resnet34['name']
        except:
            weights = get_file(resnet34['name'], resnet34['url'],
                               cache_subdir='models', md5_hash=resnet34['md5'])
    else:
        try:
            weights = WEIGHTS_DIR+resnet34_notop['name']
        except:
            weights = get_file(resnet34_notop['name'], resnet34_notop['url'],
                               cache_subdir='models', md5_hash=resnet34_notop['md5'])
    model.load_weights(weights,skip_mismatch=skip, by_name=skip)
    return model


def pretrain_vgg16(model, top=True):
    weights = None
    if top == True:
        try:
            weights = WEIGHTS_DIR+vgg16['name']
        except:
            keras.utils.get_file(
                vgg16['url'].split("/")[-1], vgg16['url'])
    else:
        try:
            weights = WEIGHTS_DIR+vgg16_notop['name']
        except:
            keras.utils.get_file(
                vgg16_notop['url'].split("/")[-1], vgg16_notop['url'])
    model.load_weights(weights)
    return model
