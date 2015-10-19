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

from train_utils import (
        ResponseNormalizationLayer,
        make_identity_transform,
        normalize_patch,
        load_dataset,
        load_identifier_eval,
        shuffle_dataset,
        train_epoch,
        load_whole_image,
        dataset_loc,
        parameter_analysis,
        display_losses)

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

def build_segmenter():
    inp = ll.InputLayer(shape=(None, 3, None, None), name='input')
    conv1 = ll.Conv2DLayer(inp, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv1')
    conv2 = ll.Conv2DLayer(conv1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv2')
    conv3 = ll.Conv2DLayer(conv2, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv3')
    conv4 = ll.Conv2DLayer(conv3, num_filters=16, filter_size=(3,3), pad='same', W=Orthogonal(), nonlinearity=rectify, name='conv4')

    # our output layer is also convolutional, remember that our Y is going to be the same exact size as the
    conv_final = ll.Conv2DLayer(conv4, num_filters=3, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv_final', nonlinearity=linear)
    # we need to reshape it to be a (batch*n*m x 3), i.e. unroll s.t. the feature dimension is preserved
    softmax = Softmax4D(conv_final, name='4dsoftmax')

    return softmax

def loss_iter(segmenter, update_params={}):
    X = T.tensor4()
    y = T.tensor4()
    pixel_weights = T.tensor3()

    all_layers = ll.get_all_layers(segmenter)
    imwrite_architecture(all_layers, './layer_rep.png')
    predicted_mask_train = ll.get_output(segmenter, X)
    predicted_mask_valid = ll.get_output(segmenter, X, deterministic=True)

    pixel_weights_1d = pixel_weights.flatten(ndim=1)
    losses = lambda pred: T.mean(crossentropy_flat(pred, y + 1e-7) * pixel_weights_1d)

    decay = 0
    reg = regularize_network_params(segmenter, l2) * decay
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = losses_reg(predicted_mask_train)
    loss_train.name = 'combined_loss' # for the names
    all_params = ll.get_all_params(segmenter)
    grads = T.grad(loss_train, all_params, add_names=True)
    #updates = adam(grads, all_params, **update_params)
    updates = adam(grads, all_params, **update_params)

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X, y, pixel_weights], [loss_train, losses(predicted_mask_train)] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X, y, pixel_weights], losses(predicted_mask_valid))
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y
    # we need to put out the pixel weight map as well

    patches = normalize_patch(dataset[0])
    #patches = np.array(patches.reshape(-1, patches.shape[3], patches.shape[1], patches.shape[2]), dtype='float32')
    patches = np.array(patches.swapaxes(1,3), dtype='float32')
    # fake patches for debugging
    #patches = np.zeros(patches.shape, dtype=patches.dtype)
    #patches += np.random.rand(*(patches.shape))

    # bleh no dimshuffle in numpy
    #labels = np.array(dataset[1].reshape(-1, dataset[1].shape[3], dataset[1].shape[1], dataset[1].shape[2]), dtype='float32')
    labels = np.array(dataset[1].swapaxes(1,3), dtype='float32')
    # a test
    #np.random.shuffle(labels)
    #print(np.argmax(labels, axis=1))
    pixel_weights = np.ones((labels.shape[0], labels.shape[2], labels.shape[3]), dtype='float32')
    # theoretically there would be 1/32 edge to non-edge ratio
    # but the way it's sampled doesn't really lead to that, it's more like a 1/50, and they're important
    pixel_weights[np.argmax(labels, axis=1) == 0] = 1.

    return shuffle_dataset({'X':patches, 'y':labels, 'pixelw':pixel_weights})

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size')
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs")
    options, args = parser.parse_args()
    if options.test:
        test_data = np.zeros((1,3,32,32), dtype='float32')
        test_data += np.random.rand(1, 3, 32, 32)
        test_y = np.zeros((1,3,32,32),dtype='float32')
        test_y[:,0,:,:] = np.ones((1,1,32,32))

        X = T.tensor4()
        y = T.tensor4()
        network = build_segmenter()
        segmenter = ll.get_output(network, X)
        loss = crossentropy_flat(segmenter, y)
        output = loss.eval({X:test_data, y:test_y})
        print(output)
        sys.exit(0)
    dset_name = options.dataset
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    print("Loading dataset")
    tic = time.time()
    dset = load_dataset(join(dataset_loc, "Flukes/patches/%s" % dset_name))
    toc = time.time() - tic
    dset = {section:preproc_dataset(dset[section]) for section in dset}
    epoch_losses = []
    batch_losses = []
    segmenter = build_segmenter()
    model_path = join(dataset_loc, "Flukes/patches/%s/model.pkl" % dset_name)
    if options.resume and exists(model_path):
        with open(model_path, 'r') as f:
            params = pickle.load(f)
        ll.set_all_param_values(segmenter, params)
    #iter_funcs = loss_iter(segmenter, update_params={'learning_rate':.01})
    iter_funcs = loss_iter(segmenter, update_params={})
    best_params = ll.get_all_param_values(segmenter)
    best_val_loss = np.inf
    for epoch in range(n_epochs):
        tic = time.time()
        print("Epoch %d" % (epoch))
        loss = train_epoch(iter_funcs, dset, batch_size=batch_size)
        epoch_losses.append(loss['train_loss'])
        batch_losses.append(loss['all_train_loss'])
        # shuffle training set
        dset['train'] = shuffle_dataset(dset['train'])
        toc = time.time() - tic
        print("Train loss (reg): %0.3f\nTrain loss: %0.3f\nValid loss: %0.3f" %
                (loss['train_reg_loss'],loss['train_loss'],loss['valid_loss']))
        if loss['valid_loss'] < best_val_loss:
            best_params = ll.get_all_param_values(segmenter)
            best_val_loss = loss['valid_loss']
            print("New best validation loss!")
        print("Took %0.2f seconds" % toc)
    batch_losses = list(chain(*batch_losses))
    losses = {}
    losses['batch'] = batch_losses
    losses['epoch'] = epoch_losses
    parameter_analysis(segmenter)
    display_losses(losses, n_epochs, batch_size, dset['train']['X'].shape[0])

    # TODO: move to train_utils and add way to load up previous model
    with open(join(dataset_loc, "Flukes/patches/%s/model.pkl" % dset_name), 'w') as f:
        pickle.dump(best_params, f)









