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
        get_img_norm_consts,
        load_dataset,
        load_identifier_eval,
        shuffle_dataset,
        train_epoch,
        load_whole_image,
        dataset_loc,
        parameter_analysis,
        display_losses,
        build_vgg16)

def build_network():



def loss_iter(classifier, update_params={}):
    X = T.tensor4()
    y = T.tensor4()
    #pixel_weights = T.tensor3()

    all_layers = ll.get_all_layers(classifier)
    imwrite_architecture(all_layers, join(dataset_loc, 'Flukes/humpnet/layer_rep.png'))
    predicted_class_train = ll.get_output(classifier, X)
    predicted_class_valid = ll.get_output(classifier, X, deterministic=True)

    accuracy = lambda pred: T.mean(T.eq(T.argmax(pred, axis=1), T.argmax(y, axis=1)))

    losses = lambda pred: T.mean(crossentropy_flat(pred + 1e-7, y + 1e-7))

    decay = 0.0001
    reg = regularize_network_params(classifier, l2) * decay
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = losses_reg(predicted_class_train)
    loss_train.name = 'combined_loss' # for the names
    all_params = ll.get_all_params(classifier)
    # we actually only want the last layer's gradient
    # and even then we only want the slice corresponding to the 1000'th class (humpbacks)

    grads = T.grad(loss_train, all_params, add_names=True)
    #updates = adam(grads, all_params, **update_params)
    updates = adam(grads, all_params, **update_params)
    acc_train = accuracy(predicted_class_train)
    acc_valid = accuracy(predicted_class_valid)

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X, y], [loss_train, losses(predicted_class_train), acc_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X, y], [losses(predicted_class_valid), acc_valid])
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y
    # we need to put out the pixel weight map as well

    patches = dataset[0]
    #patches = np.array(patches.reshape(-1, patches.shape[3], patches.shape[1], patches.shape[2]), dtype='float32')
    patches = np.array(patches.swapaxes(1,3), dtype='float32')

    print(np.average(patches, axis=(0,2,3)))
    print(np.std(patches, axis=(0,2,3)))
    labels = dataset[1]

    return shuffle_dataset({'X':patches, 'y':labels,})

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    #parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size', default=32)
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs", default=1)
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
    dset = load_dataset(join(dataset_loc, "Flukes/humpnet"))
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    epoch_losses = []
    batch_losses = []
    segmenter = build_segmenter_vgg()
    model_path = join(dataset_loc, "Flukes/humpnet/model.pkl")
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
        print("Train acc: %0.3f\nValid acc: %0.3f" % (loss['train_acc'], loss['valid_acc']))
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
    with open(join(dataset_loc, "Flukes/humpnet/model.pkl"), 'w') as f:
        pickle.dump(best_params, f)



