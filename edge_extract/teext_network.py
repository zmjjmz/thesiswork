from __future__ import division

from itertools import chain, product
from functools import partial
import random

import cPickle as pickle
from os.path import join, exists
import time
from sklearn.utils import shuffle
import sys
import numpy as np
from ibeis_cnn.draw_net import imwrite_architecture
from optparse import OptionParser

import lasagne.layers as ll
from lasagne.nonlinearities import linear, softmax, sigmoid, rectify
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam, nesterov_momentum
from lasagne.init import Orthogonal, Constant
from lasagne.regularization import l2, regularize_network_params
import theano.tensor as T
import theano
from theano.compile.nanguardmode import NanGuardMode
from da import transform
import utool as ut

from train_utils import (
        ResponseNormalizationLayer,
        make_identity_transform,
        normalize_patch,
        get_img_norm_consts,
        load_dataset,
        load_identifier_eval,
        shuffle_dataset,
        train_epoch,
        load_whole_image,
        dataset_loc,
        parameter_analysis,
        display_losses,
        build_vgg16_seg,
        make_identity_transform,)

class Softmax4D(ll.Layer):
    def get_output_for(self, input, **kwargs):
        si = input.reshape((input.shape[0], input.shape[1], -1))
        shp = (si.shape[0], 1, si.shape[2])
        exp = T.exp(si - si.max(axis=1).reshape(shp))
        softmax_expression = (exp / (exp.sum(axis=1).reshape(shp) + 1e-7) ).reshape(input.shape)
        return softmax_expression

def crossentropy_flat(pred, true):
    # basically we have a distribution output that's in the shape batch, prob, h, w
    # it doesn't look like we can apply the nnet categorical cross entropy easily on a tensor4
    # so we'll have to flatten it out to a tensor2, which is a pain in the ass but easily done

    pred2 = pred.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    true2 = true.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)

    return T.nnet.categorical_crossentropy(pred2, true2)


def build_segmenter_simple():
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(7,7), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2')
    conv3 = ll.Conv2DLayer(conv2, num_filters=128, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=64, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4')
    conv5 = ll.Conv2DLayer(conv4, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv5')
    conv6 = ll.Conv2DLayer(conv5, num_filters=16, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv6')

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(conv6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return [softmax]

def build_segmenter_simple_absurd():
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    n_layers = 64 # should get a 128 x 128 receptive field
    layers = [inp]
    for i in range(n_layers):
        layers.append(ll.Conv2DLayer(layers[-1], num_filters=8, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv%d' % (i+1)))
        layers.append(ll.BatchNormLayer(layers[-1], name='bn%i' % (i+1)))

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(layers[-1], num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return [softmax]

def build_segmenter_simple_absurd_res():
    sys.setrecursionlimit(1500)
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    n_layers = 64 # should get a 128 x 128 receptive field
    layers = [inp]
    for i in range(n_layers):
        # every 2 layers, add a skip connection
        layers.append(ll.Conv2DLayer(layers[-1], num_filters=8, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear, name='conv%d' % (i+1)))
        layers.append(ll.BatchNormLayer(layers[-1], name='bn%i' % (i+1)))
        if (i % 2 == 0) and (i != 0):
            layers.append(ll.ElemwiseSumLayer([layers[-1], # prev layer
                                              layers[-6],] # 3 actual layers per block, skip the previous block
                                              ))
        layers.append(ll.NonlinearityLayer(layers[-1], nonlinearity=rectify))

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(layers[-1], num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return [softmax]




def build_segmenter_upsample():
    # downsample down to a small region, then upsample all the way back up
    # Note: w/o any learning on the upsampler, we're limited in how far we can downsample
    # there will always be an error signal unless the loss fn is run on downsampled targets...
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    bn1 = ll.BatchNormLayer(conv1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    bn2 = ll.BatchNormLayer(conv2, name='bn2')
    mp1 = ll.MaxPool2DLayer(bn2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    bn3 = ll.BatchNormLayer(conv3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    bn4 = ll.BatchNormLayer(conv4, name='bn4')
    mp2 = ll.MaxPool2DLayer(bn4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    bn5 = ll.BatchNormLayer(conv5, name='bn5')
    conv6 = ll.Conv2DLayer(bn5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    bn6 = ll.BatchNormLayer(conv6, name='bn6')
    mp3 = ll.MaxPool2DLayer(bn6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    bn7 = ll.BatchNormLayer(conv7, name='bn7')
    conv8 = ll.Conv2DLayer(bn7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    bn8 = ll.BatchNormLayer(conv8, name='bn8')
    # f 68 s 8
    # now start the upsample
    up = ll.Upscale2DLayer(bn8, 8, name='upsample_8x')
    conv_f = ll.Conv2DLayer(up, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear, name='conv_final')
    softmax = Softmax4D(conv_f, name='4dsoftmax')
    return [softmax]


def build_segmenter_jet():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    bn1 = ll.BatchNormLayer(conv1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    bn2 = ll.BatchNormLayer(conv2, name='bn2')
    mp1 = ll.MaxPool2DLayer(bn2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    bn3 = ll.BatchNormLayer(conv3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    bn4 = ll.BatchNormLayer(conv4, name='bn4')
    mp2 = ll.MaxPool2DLayer(bn4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    bn5 = ll.BatchNormLayer(conv5, name='bn5')
    conv6 = ll.Conv2DLayer(bn5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    bn6 = ll.BatchNormLayer(conv6, name='bn6')
    mp3 = ll.MaxPool2DLayer(bn6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    bn7 = ll.BatchNormLayer(conv7, name='bn7')
    conv8 = ll.Conv2DLayer(bn7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    bn8 = ll.BatchNormLayer(conv8, name='bn8')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)
    conv_f8 = ll.Conv2DLayer(bn8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')
    up8 = ll.Upscale2DLayer(softmax_8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX 8 AND PRED ON CONV 6
    softmax_4up = ll.Upscale2DLayer(softmax_8, 2, name='upsample_4x_pre') # 4x downsample
    conv_f6 = ll.Conv2DLayer(bn6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f6, name='4dsoftmax_4x') # 4x downsample
    softmax_4_merge = ll.ElemwiseSumLayer([softmax_4, softmax_4up], coeffs=0.5, name='softmax_4_merge')

    up4 = ll.Upscale2DLayer(softmax_4_merge, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_4_MERGE AND CONV 4
    softmax_2up = ll.Upscale2DLayer(softmax_4_merge, 2, name='upsample_2x_pre') # 2x downsample
    conv_f4 = ll.Conv2DLayer(bn4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f4, name='4dsoftmax_2x')
    softmax_2_merge = ll.ElemwiseSumLayer([softmax_2, softmax_2up], coeffs=0.5, name='softmax_2_merge')

    up2 = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample

    return [up8, up4, up2]

def build_segmenter_jet_2():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    # this jet will have another conv layer in the final upsample
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    bn1 = ll.BatchNormLayer(conv1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    bn2 = ll.BatchNormLayer(conv2, name='bn2')
    mp1 = ll.MaxPool2DLayer(bn2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    bn3 = ll.BatchNormLayer(conv3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    bn4 = ll.BatchNormLayer(conv4, name='bn4')
    mp2 = ll.MaxPool2DLayer(bn4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    bn5 = ll.BatchNormLayer(conv5, name='bn5')
    conv6 = ll.Conv2DLayer(bn5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    bn6 = ll.BatchNormLayer(conv6, name='bn6')
    mp3 = ll.MaxPool2DLayer(bn6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    bn7 = ll.BatchNormLayer(conv7, name='bn7')
    conv8 = ll.Conv2DLayer(bn7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    bn8 = ll.BatchNormLayer(conv8, name='bn8')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)
    conv_f8 = ll.Conv2DLayer(bn8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')
    up8 = ll.Upscale2DLayer(softmax_8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX 8 AND PRED ON CONV 6
    softmax_4up = ll.Upscale2DLayer(softmax_8, 2, name='upsample_4x_pre') # 4x downsample
    conv_f6 = ll.Conv2DLayer(bn6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f6, name='4dsoftmax_4x') # 4x downsample
    softmax_4_merge = ll.ElemwiseSumLayer([softmax_4, softmax_4up], coeffs=0.5, name='softmax_4_merge')

    up4 = ll.Upscale2DLayer(softmax_4_merge, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_4_MERGE AND CONV 4
    softmax_2up = ll.Upscale2DLayer(softmax_4_merge, 2, name='upsample_2x_pre') # 2x downsample
    conv_f4 = ll.Conv2DLayer(bn4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f4, name='4dsoftmax_2x')
    softmax_2_merge = ll.ElemwiseSumLayer([softmax_2, softmax_2up], coeffs=0.5, name='softmax_2_merge')

    up2 = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample

    ## COMBINE BY UPSAMPLING SOFTMAX_2_MERGE AND CONV 2
    softmax_1up = ll.Upscale2DLayer(softmax_2_merge, 2, name='upsample_1x_pre') # 1x downsample (i.e. no downsample)
    conv_f2 = ll.Conv2DLayer(bn2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_1xpred')

    softmax_1 = Softmax4D(conv_f2, name='4dsoftmax_1x')
    softmax_1_merge = ll.ElemwiseSumLayer([softmax_1, softmax_1up], coeffs=0.5, name='softmax_1_merge')

    # this is where up1 would go but that doesn't make any sense
    return [up8, up4, up2, softmax_1_merge]


def build_segmenter_jet_preconv():
    # downsample down to a small region, then upsample all the way back up, using jet architecture
    # recreate basic FCN-8s structure (though more aptly 1s here since we upsample back to the original input size)
    # this jet will have another conv layer in the final upsample
    # difference here is that instead of combining softmax layers in the jet, we'll upsample before the conv_f* layer
    # this will certainly make the model slower, but should give us better predictions...
    # The awkward part here is combining the intermediate conv layers when they have different filter shapes
    # We could:
    #   concat them
    #   have intermediate conv layers that bring them to the shape needed then merge them
    # in the interests of speed we'll just concat them, though we'll have a ton of filters at the end
    inp = ll.InputLayer(shape=(None, 1, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_1')
    bn1 = ll.BatchNormLayer(conv1, name='bn1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1_2')
    bn2 = ll.BatchNormLayer(conv2, name='bn2')
    mp1 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp1') # 2x downsample
    conv3 = ll.Conv2DLayer(mp1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_1')
    bn3 = ll.BatchNormLayer(conv3, name='bn3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2_2')
    bn4 = ll.BatchNormLayer(conv4, name='bn4')
    mp2 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp2') # 4x downsample
    conv5 = ll.Conv2DLayer(mp2, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_1')
    bn5 = ll.BatchNormLayer(conv5, name='bn5')
    conv6 = ll.Conv2DLayer(conv5, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3_2')
    bn6 = ll.BatchNormLayer(conv6, name='bn6')
    mp3 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp3') # 8x downsample
    conv7 = ll.Conv2DLayer(mp3, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_1')
    bn7 = ll.BatchNormLayer(conv7, name='bn7')
    conv8 = ll.Conv2DLayer(conv7, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4_2')
    bn8 = ll.BatchNormLayer(conv8, name='bn8')
    # f 68 s 8
    # now start the upsample
    ## FIRST UPSAMPLE PREDICTION (akin to FCN-32s)

    up8 = ll.Upscale2DLayer(bn8, 8, name='upsample_8x') # take loss here, 8x upsample from 8x downsample
    conv_f8 = ll.Conv2DLayer(up8, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_8xpred')
    softmax_8 = Softmax4D(conv_f8, name='4dsoftmax_8x')

    ## COMBINE BY UPSAMPLING CONV 8 AND CONV 6
    conv_8_up2 = ll.Upscale2DLayer(bn8, 2, name='upsample_c8_2') # 4x downsample
    concat_c8_c6 = ll.ConcatLayer([conv_8_up2, bn6], axis=1, name='concat_c8_c6')
    up4 = ll.Upscale2DLayer(concat_c8_c6, 4, name='upsample_4x') # take loss here, 4x upsample from 4x downsample
    conv_f4 = ll.Conv2DLayer(up4, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_4xpred')
    softmax_4 = Softmax4D(conv_f4, name='4dsoftmax_4x') # 4x downsample

    ## COMBINE BY UPSAMPLING CONCAT_86 AND CONV 4
    concat_86_up2 = ll.Upscale2DLayer(concat_c8_c6, 2, name='upsample_concat_86_2') # 2x downsample
    concat_ct86_c4 = ll.ConcatLayer([concat_86_up2, bn4], axis=1, name='concat_ct86_c4')

    up2 = ll.Upscale2DLayer(concat_ct86_c4, 2, name='upsample_2x') # final loss here, 2x upsample from a 2x downsample
    conv_f2 = ll.Conv2DLayer(up2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_2xpred')

    softmax_2 = Softmax4D(conv_f2, name='4dsoftmax_2x')


    ## COMBINE BY UPSAMPLING CONCAT_864 AND CONV 2
    concat_864_up2 = ll.Upscale2DLayer(concat_ct86_c4, 2, name='upsample_concat_86_2') # no downsample
    concat_864_c2 = ll.ConcatLayer([concat_864_up2, bn2], axis=1, name='concat_ct864_c2')
    conv_f1 = ll.Conv2DLayer(concat_864_c2, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=linear,
                             name='conv_1xpred')

    softmax_1 = Softmax4D(conv_f1, name='4dsoftmax_1x')

    # this is where up1 would go but that doesn't make any sense
    return [softmax_8, softmax_4, softmax_2, softmax_1]






def build_segmenter_vgg():
    vgg_net = build_vgg16_seg()
    conv3 = ll.Conv2DLayer(vgg_net['conv4_3'], num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4')
    conv_final = ll.Conv2DLayer(conv4, num_filters=3, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    softmax = Softmax4D(conv_final, name='4dsoftmax')
    return softmax



def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y
    # we need to put out the pixel weight map as well


    # assume patches are already greyscale
    patches = np.expand_dims(dataset[0].astype(np.float32),axis=1)
    labels = np.array(np.transpose(dataset[1].swapaxes(1,3),(0,1,3,2)), dtype='float32')
    pixel_weights = np.ones((labels.shape[0], labels.shape[2], labels.shape[3]), dtype='float32')
    # theoretically there would be 1/32 edge to non-edge ratio
    # but the way it's sampled doesn't really lead to that, it's more like a 1/50, and they're important
    pixel_weights[np.argmax(labels, axis=1) == 0] = 1e1

    return shuffle_dataset({'X':patches, 'y':labels, 'pixelw':pixel_weights})

def loss_iter(segmenter, update_params={}):
    X = T.tensor4()
    y = T.tensor4()
    pixel_weights = T.tensor3()

    final_pred_layer = segmenter[-1]
    all_layers = ll.get_all_layers(segmenter)
    imwrite_architecture(all_layers, './layer_rep.png')
    # assume we get a list of predictions (e.g. for jet architecture, but should work w/just one pred)
    # another assumption (which must hold when the network is being made)
    # the last prediction layer is a) the end of the network and b) what we ultimately care about
    # however the other prediction layers will be incorporated into the training loss
    predicted_masks_train = ll.get_output(segmenter, X)
    predicted_mask_valid = ll.get_output(final_pred_layer, X, deterministic=True)

    thresh = 0.5
    accuracy = lambda pred: T.mean(T.eq(T.argmax(pred, axis=1), T.argmax(y, axis=1)))
    true_pos = lambda pred: T.sum((pred[:,0,:,:] > thresh) * (y[:,0,:,:] > thresh))
    false_pos = lambda pred: T.sum((pred[:,0,:,:] > thresh) - (y[:,0,:,:] > thresh))
    precision = lambda pred: (true_pos(pred) / (true_pos(pred) + false_pos(pred)))

    pixel_weights_1d = pixel_weights.flatten(ndim=1)
    losses = lambda pred: T.mean(crossentropy_flat(pred + 1e-7, y + 1e-7) * pixel_weights_1d)

    decay = 0.0001
    reg = regularize_network_params(final_pred_layer, l2) * decay
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = T.sum([losses_reg(mask) for mask in predicted_masks_train])
    loss_train.name = 'CE' # for the names
    #all_params = list(chain(*[ll.get_all_params(pred) for pred in segmenter]))
    all_params = ll.get_all_params(segmenter, trainable=True) # this should work with multiple 'roots'
    grads = T.grad(loss_train, all_params, add_names=True)
    updates = adam(grads, all_params)
    #updates = nesterov_momentum(grads, all_params, update_params['l_r'], momentum=update_params['momentum'])
    acc_train = accuracy(predicted_masks_train[-1])
    acc_valid = accuracy(predicted_mask_valid)
    prec_train = precision(predicted_masks_train[-1])
    prec_valid = precision(predicted_mask_valid)

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X, y, pixel_weights], [loss_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X, y, pixel_weights], [losses(predicted_mask_valid), losses_reg(predicted_mask_valid), prec_valid])
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def augmenting_batch_loader(dataset, bslice, sec):
    if sec != 'train':
        return [dataset[s][bslice] for s in ['X','y','pixelw']]
    else:
        Xaug, yaug = transform(dataset['X'][bslice], dataset['y'][bslice], transform_y=True, y_coords=False)
        if bslice.start == 0:
            ut.save_cPkl(join(dataset_loc, "Flukes/patches/sample_aug.pkl"), (dataset['X'][bslice], dataset['y'][bslice], Xaug, yaug))
        return [Xaug, yaug, dataset['pixelw'][bslice]]

def nonaugmenting_batch_loader(dataset, bslice, sec):
    return [dataset[s][bslice] for s in ['X','y','pixelw']]

SEGMENTERS = {
    'simple':build_segmenter_simple,
    'upsample':build_segmenter_upsample,
    'jet':build_segmenter_jet,
    'jet2':build_segmenter_jet_2,
    'jet_preconv':build_segmenter_jet_preconv,
    'absurd':build_segmenter_simple_absurd,
    'absurd_res':build_segmenter_simple_absurd_res,
}

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size', default=32)
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs", default=1)
    parser.add_option("-m", "--model", action="store", type="string", dest="model", default="jet2")
    options, args = parser.parse_args()
    if options.test:
        print("No test stuff defined")
        sys.exit(0)
    dset_name = options.dataset
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    model_type = options.model
    if model_type not in SEGMENTERS:
        print("Invalid model specified: %s" % model_type)
        print("Valid models are: %r" % SEGMENTERS.keys())
        sys.exit(1)
    print("Loading dataset")
    tic = time.time()
    dset = load_dataset(join(dataset_loc, "Flukes/patches/%s" % dset_name), normalize_method='zscore')
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    epoch_losses = []
    batch_losses = []
    segmenter = SEGMENTERS[model_type]()
    model_path = join(dataset_loc, "Flukes/patches/%s/model%s.pkl" % (dset_name, model_type))
    if options.resume and exists(model_path):
        params = ut.load_cPkl(model_path)
        ll.set_all_param_values(segmenter, params)
    #iter_funcs = loss_iter(segmenter, update_params={'learning_rate':.01})
    lr = theano.shared(np.array(0.010, dtype=np.float32))
    momentum_params = {'l_r':lr, 'momentum':0.9}
    iter_funcs = loss_iter(segmenter, update_params=momentum_params)
    best_params = ll.get_all_param_values(segmenter)
    best_val_loss = np.inf
    layer_names = [p.name for p in ll.get_all_params(segmenter, trainable=True)]
    save_model = True
    try:
        for epoch in range(n_epochs):
            tic = time.time()
            print("Epoch %d" % (epoch))
            loss = train_epoch(iter_funcs, dset, batch_size, augmenting_batch_loader, layer_names=layer_names)
            epoch_losses.append(loss['train_loss'])
            batch_losses.append(loss['all_train_loss'])
            # shuffle training set
            dset['train'] = shuffle_dataset(dset['train'])
            toc = time.time() - tic
            print("Learning rate: %0.5f" % momentum_params['l_r'].get_value())
            print("Train loss (reg): %0.3f\nTrain loss: %0.3f\nValid loss: %0.3f" %
                    (loss['train_reg_loss'],loss['train_loss'],loss['valid_loss']))
            print("Train Pixel Precision @0.5: %s\nValid Pixel Precision @0.5: %s" % (loss['train_acc'], loss['valid_acc']))
            if loss['valid_loss'] < best_val_loss:
                best_params = ll.get_all_param_values(segmenter)
                best_val_loss = loss['valid_loss']
                print("New best validation loss!")
            print("Took %0.2f seconds" % toc)
    except KeyboardInterrupt:
        print("Training interrupted, save model? y/n")
        confirm = raw_input().rstrip()
        if confirm == 'n':
            print("Not saving model")
            save_model = False

    batch_losses = list(chain(*batch_losses))
    losses = {}
    losses['batch'] = batch_losses
    losses['epoch'] = epoch_losses
    parameter_analysis(segmenter)
    if save_model:
        ut.save_cPkl(model_path, best_params)
        display_losses(losses, n_epochs, batch_size, dset['train']['X'].shape[0])
