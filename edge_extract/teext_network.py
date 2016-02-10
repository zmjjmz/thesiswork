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
    conv1 = ll.Conv2DLayer(inp, num_filters=4, filter_size=(7,7), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=8, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2')
    conv3 = ll.Conv2DLayer(conv2, num_filters=16, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=8, filter_size=(5,5), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4')
    conv5 = ll.Conv2DLayer(conv4, num_filters=8, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv5')
    conv6 = ll.Conv2DLayer(conv5, num_filters=4, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv6')

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(conv6, num_filters=2, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return softmax


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
    pixel_weights[np.argmax(labels, axis=1) == 0] = 1e2

    return shuffle_dataset({'X':patches, 'y':labels, 'pixelw':pixel_weights})

def loss_iter(segmenter, update_params={}):
    X = T.tensor4()
    y = T.tensor4()
    pixel_weights = T.tensor3()

    all_layers = ll.get_all_layers(segmenter)
    imwrite_architecture(all_layers, './layer_rep.png')
    predicted_mask_train = ll.get_output(segmenter, X)
    predicted_mask_valid = ll.get_output(segmenter, X, deterministic=True)

    thresh = 0.5
    accuracy = lambda pred: T.mean(T.eq(T.argmax(pred, axis=1), T.argmax(y, axis=1)))
    true_pos = lambda pred: T.sum((pred[:,0,:,:] > thresh) * (y[:,0,:,:] > thresh))
    false_pos = lambda pred: T.sum((pred[:,0,:,:] > thresh) - (y[:,0,:,:] > thresh))
    precision = lambda pred: (true_pos(pred) / (true_pos(pred) + false_pos(pred)))

    pixel_weights_1d = pixel_weights.flatten(ndim=1)
    losses = lambda pred: T.mean(crossentropy_flat(pred + 1e-7, y + 1e-7) * pixel_weights_1d)

    decay = 0.0001
    reg = regularize_network_params(segmenter, l2) * decay
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = losses_reg(predicted_mask_train)
    loss_train.name = 'CE' # for the names
    all_params = ll.get_all_params(segmenter)
    grads = T.grad(loss_train, all_params, add_names=True)
    updates = nesterov_momentum(grads, all_params, update_params['l_r'], momentum=update_params['momentum'])
    acc_train = accuracy(predicted_mask_train)
    acc_valid = accuracy(predicted_mask_valid)
    prec_train = precision(predicted_mask_train)
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
        Xaug, yaug = transform(dataset['X'][bslice], dataset['y'][bslice]*dataset['X'][bslice].shape[-1], transform_y=True)
        yaug /= dataset['X'][bslice].shape[-1]
        #with open(join(dataset_loc, "Flukes/kpts/sample_aug.pkl"), 'w') as f:
        #    pickle.dump((dataset['X'][bslice], dataset['y'][bslice], Xaug, yaug), f)
        return [Xaug, yaug, dataset['pixelw'][bslice]]

def nonaugmenting_batch_loader(dataset, bslice, sec):
    return [dataset[s][bslice] for s in ['X','y','pixelw']]

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size', default=32)
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs", default=1)
    options, args = parser.parse_args()
    if options.test:
        print("No test stuff defined")
        sys.exit(0)
    dset_name = options.dataset
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    print("Loading dataset")
    tic = time.time()
    dset = load_dataset(join(dataset_loc, "Flukes/patches/%s" % dset_name), normalize_method='zscore')
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    epoch_losses = []
    batch_losses = []
    kp_extractor = build_segmenter_simple()
    model_path = join(dataset_loc, "Flukes/patches/%s/model.pkl" % dset_name)
    if options.resume and exists(model_path):
        with open(model_path, 'r') as f:
            params = pickle.load(f)
        ll.set_all_param_values(kp_extractor, params)
    #iter_funcs = loss_iter(kp_extractor, update_params={'learning_rate':.01})
    lr = theano.shared(np.array(0.010, dtype=np.float32))
    momentum_params = {'l_r':lr, 'momentum':0.9}
    iter_funcs = loss_iter(kp_extractor, update_params=momentum_params)
    best_params = ll.get_all_param_values(kp_extractor)
    best_val_loss = np.inf
    layer_names = [p.name for p in ll.get_all_params(kp_extractor, trainable=True)]
    for epoch in range(n_epochs):
        tic = time.time()
        print("Epoch %d" % (epoch))
        loss = train_epoch(iter_funcs, dset, batch_size, nonaugmenting_batch_loader, layer_names=layer_names)
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
            best_params = ll.get_all_param_values(kp_extractor)
            best_val_loss = loss['valid_loss']
            print("New best validation loss!")
        print("Took %0.2f seconds" % toc)
    batch_losses = list(chain(*batch_losses))
    losses = {}
    losses['batch'] = batch_losses
    losses['epoch'] = epoch_losses
    parameter_analysis(kp_extractor)
    display_losses(losses, n_epochs, batch_size, dset['train']['X'].shape[0])

    # TODO: move to train_utils and add way to load up previous model
    with open(join(dataset_loc, "Flukes/patches/%s/model.pkl" % dset_name), 'w') as f:
        pickle.dump(best_params, f)
