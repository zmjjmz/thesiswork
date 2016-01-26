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
        build_vgg16_seg)

def build_kpextractor64():
    inp = ll.InputLayer(shape=(None, 1, 64, 64), name='input')
    # we're going to build something like what Daniel Nouri made for Facial Keypoint detection for a base reference
    # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=8, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 32 x 32
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 16 x 16
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 8 x 8
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    # larger max pool to reduce parameters in the FC layer
    mp4 = ll.MaxPool2DLayer(conv4, 4, stride=4) # now down to 2x2
    bn4 = ll.BatchNormLayer(mp4)
    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(bn4, num_units=128, nonlinearity=rectify)
    bn5 = ll.BatchNormLayer(fc1)
    #dp1 = ll.DropoutLayer(bn1, p=0.5)
    fc2 = ll.DenseLayer(bn5, num_units=64, nonlinearity=rectify)
    #dp2 = ll.DropoutLayer(fc2, p=0.5)
    bn6 = ll.BatchNormLayer(fc2)
    out = ll.DenseLayer(bn6, num_units=6, nonlinearity=linear)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs


def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y

    imgs = np.expand_dims(dataset[0].astype(np.float32),axis=1) # add fake channel axis to greyscaled images
    imgs_T = np.transpose(imgs,axes=(0,1,3,2)) # the most insane thing I've done

    img_shape = imgs_T.shape[-2:]
    # assume img_shape is y, x

    # scale the points
    pts = dataset[1].astype(np.float32)
    scale_mat = np.array([img_shape[1], img_shape[0]] * pts.shape[1], dtype=np.float32).reshape(pts.shape[1], 2)
    pts_scaled = (pts - scale_mat / 2) / (scale_mat / 2)

    return shuffle_dataset({'X':imgs_T, 'y':pts})



def loss_iter(kpextractor, update_params={}):
    X = T.tensor4()
    y = T.tensor3() # first axis is batch axis, second is point axis (left, right, notch) third is x, y

    all_layers = ll.get_all_layers(kpextractor)
    imwrite_architecture(all_layers, './layer_rep_kpext.png')
    predicted_points_train = ll.get_output(kpextractor, X)
    predicted_points_valid = ll.get_output(kpextractor, X, deterministic=True)

    rmse = lambda pred: T.sqrt(T.mean((pred - y)**2, axis=(1,2)))
    eucl = lambda pred: T.mean((pred - y).norm(2, axis=2), axis=1)

    # experiment: penalize the difference between the avg std of the output over the batch
    # and the avg std of the true points over the batch
    std_diff = lambda pred: T.sqrt(T.mean((T.std(pred,axis=0) - T.std(y,axis=0))**2))
    losses = lambda pred: T.mean(eucl(pred))
    decay = 1e-3
    reg = regularize_network_params(kpextractor, l2) * decay

    #predT_p = theano.printing.Print()(T.mean(predicted_points_train,axis=0))
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = losses_reg(predicted_points_train)
    loss_train.name = 'scaled_rmse' # assuming that the y values are scaled to -1, 1
    all_params = ll.get_all_params(kpextractor, trainable=True)
    grads = T.grad(loss_train, all_params, add_names=True)
    #updates = adam(grads, all_params, **update_params)
    updates = adam(grads, all_params, **update_params)

    # calculate and report average pixel distance
    sizey, sizex = all_layers[0].shape[-2:]
    scale_vec = np.array([sizex, sizey]*3).reshape(3,2)
    scaled_y = (y * scale_vec + scale_vec) / 2
    scaled_pred = lambda pred: (pred * scale_vec + scale_vec) / 2
    #avg_pix_dist = lambda pred: T.mean((scaled_pred(pred) - scaled_y).norm(2, axis=2),axis=0)
    avg_pix_dist = lambda pred: T.mean((pred - y).norm(2, axis=2),axis=0)


    pix_dist_train = avg_pix_dist(predicted_points_train)
    pix_dist_valid = avg_pix_dist(predicted_points_valid)

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X, y], [loss_train,
                                          losses(predicted_points_train),
                                          pix_dist_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X, y], [losses(predicted_points_valid),
                                          pix_dist_valid])
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size', default=32)
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs", default=1)
    options, args = parser.parse_args()
    if options.test:
        test_data = np.random.rand(5, 1, 64, 64).astype(np.float32)
        X = T.tensor4()
        network = build_kpextractor64()
        kpextractor = ll.get_output(network, X)
        output = kpextractor.eval({X:test_data})
        print(output)
        sys.exit(0)
    dset_name = options.dataset
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    print("Loading dataset")
    tic = time.time()
    dset = load_dataset(join(dataset_loc, "Flukes/kpts/%s" % dset_name), normalize_method='meansub')
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    epoch_losses = []
    batch_losses = []
    kp_extractor = build_kpextractor64()
    model_path = join(dataset_loc, "Flukes/kpts/%s/model.pkl" % dset_name)
    if options.resume and exists(model_path):
        with open(model_path, 'r') as f:
            params = pickle.load(f)
        ll.set_all_param_values(kp_extractor, params)
    #iter_funcs = loss_iter(kp_extractor, update_params={'learning_rate':.01})
    iter_funcs = loss_iter(kp_extractor, update_params={})
    best_params = ll.get_all_param_values(kp_extractor)
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
        print("Train Pixel Dist: %s\nValid Pixel Dist: %s" % (loss['train_acc'], loss['valid_acc']))
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
    with open(join(dataset_loc, "Flukes/kpts/%s/model.pkl" % dset_name), 'w') as f:
        pickle.dump(best_params, f)
