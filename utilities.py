from skimage.transform import resize
import numpy as np
import pandas as pd
import six
import cv2
img_size_ori = 101
img_size_target = 128


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    # img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width, height, nClasses))

    return seg_labels


def merge_weights(pretrained, new_model, target=4):
    # put in pretrained 3 channel model
    # put in new_model with no pre-training set up with 4 channels

    for layer in pretrained.layers:  # pretrained Model and template have the same
                                    # layers, so it doesn't matter which to
                                    # iterate over.

        if layer.get_weights() != []:  # Skip input, pooling and no weights layers

            target_layer = new_model.get_layer(name=layer.name)
            # print(layer.name)
            # layer.input_shape in layers_to_modify:
            
            if layer.input_shape != target_layer.input_shape:
                starting_weights = np.array(layer.get_weights())
                # print(len(weights))
                weights = starting_weights
                print(layer.input_shape)
                print(target_layer.input_shape)
                print(weights[0].shape)

                if len(weights.shape) == 1:
                    weights = np.array(layer.get_weights())[0]

                slc = [slice(None)] * len(weights.shape)
                # for i in range(len(weights)):
                # weights[i] = np.append(weights[i], weights[i][-1])
                # print(kernels.shape)
                axis = None


                for i in range(len(weights.shape)):
                    if weights.shape[i] == 3:
                        axis = i  # Specifically looking for the last channel with 3 layers

                if axis != None:
                    print('test')
                    slc[axis] = slice(-1, None)
                    sl = weights[slc]
                    while weights.shape[axis] < target:
                        weights = np.append(weights, sl, axis=axis)
                
                
                if len(starting_weights.shape) == 1:
                    print('2')
                    new_weights = np.array(layer.get_weights())
                    new_weights[0] = weights
                    weights = new_weights

                target_layer.set_weights(weights)
                continue
            else:
                target_layer.set_weights(layer.get_weights())
    return new_model  # returns 4 channel model
