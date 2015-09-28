from __future__ import division
from itertools import chain, product
from functools import partial
import random

import cPickle as pickle
from os.path import join
import time
from sklearn.utils import shuffle
import sys
import numpy as np
from ibeis_cnn.draw_net import imwrite_architecture

import lasagne.layers as ll
from lasagne.nonlinearities import linear, softmax, sigmoid
from lasagne.objectives import binary_crossentropy
from lasagne.updates import adam, nesterov_momentum
from lasagne.init import Orthogonal, Constant
from lasagne.regularization import l2, regularize_network_params
import theano.tensor as T
import theano

from train_utils import (
        ResponseNormalizationLayer,
        make_identity_transform,
        desc_func,
        normalize_patches,
        normalize_image_idmap,
        load_dataset,
        load_identifier_eval,
        shuffle_dataset,
        train_epoch,
        load_whole_image,
        dataset_loc,
        parameter_analysis)

def build_siamese_whole(imgshape):
    # siamesse model similar to http://nbviewer.ipython.org/gist/ebenolson/40205d53a1a27ed0cc08#

    # Build first network

    # numpy image conventions are y: 0 x: 1
    inp_1 = ll.InputLayer(shape=(None, 3, imgshape[0], imgshape[1]), name='input1')
    inp_2 = ll.InputLayer(shape=(None, 3, imgshape[0], imgshape[1]), name='input2')

    # keeping it on 'same' for padding and a stride of 1 leaves the output of conv2D the same 2D shape as the input, which is convenient
    # Following Sander's guidelines here https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/cv3ib5q
    # conv2d defaults to Relu with glorot init
    conv1_1 = ll.Conv2DLayer(inp_1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv1_1')
    conv1_2 = ll.Conv2DLayer(inp_2, num_filters=128, filter_size=(3,3), pad='same', W=conv1_1.W, b=conv1_1.b, name='conv1_2')

    mpool1_1 = ll.Pool2DLayer(conv1_1, pool_size=3, mode='max', name='mpool1_1')
    mpool1_2 = ll.Pool2DLayer(conv1_2, pool_size=3, mode='max', name='mpool1_2')

    drop1_1 = ll.DropoutLayer(mpool1_1, p=0, name='drop1_1')
    drop1_2 = ll.DropoutLayer(mpool1_2, p=0, name='drop1_2')
    # now we're down to 128 / 3 = 42x42 inputs

    # 9/20: reducing the amount of filters in conv2 from 64->16 to hopefully deal with some overfitting
    # No bueno, reducing to 8 didn't help either
    conv2_1 = ll.Conv2DLayer(drop1_1, num_filters=64, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv2_1')
    conv2_2 = ll.Conv2DLayer(drop1_2, num_filters=64, filter_size=(3,3), pad='same', W=conv2_1.W, b=conv2_1.b, name='conv2_2')

    mpool2_1 = ll.Pool2DLayer(conv2_1, pool_size=3, mode='max', name='mpool2_1')
    mpool2_2 = ll.Pool2DLayer(conv2_2, pool_size=3, mode='max', name='mpool2_2')

    drop2_1 = ll.DropoutLayer(mpool2_1, p=0, name='drop2_1')
    drop2_2 = ll.DropoutLayer(mpool2_2, p=0, name='drop2_2')
    # now we're down to 42 / 3 = 14x14 inputs, and 16 of them so we'll have 14x14x16 = 3136 inputs

    conv3_1 = ll.Conv2DLayer(drop2_1, num_filters=32, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv3_1')
    conv3_2 = ll.Conv2DLayer(drop2_2, num_filters=32, filter_size=(3,3), pad='same', W=conv3_1.W, b=conv3_1.b, name='conv3_2')

    mpool3_1 = ll.Pool2DLayer(conv3_1, pool_size=3, mode='max', name='mpool3_1')
    mpool3_2 = ll.Pool2DLayer(conv3_2, pool_size=3, mode='max', name='mpool3_2')

    drop3_1 = ll.DropoutLayer(mpool3_1, p=0, name='drop3_1')
    drop3_2 = ll.DropoutLayer(mpool3_2, p=0, name='drop3_2')

    # for similarity, we'll do a few FC layers before merging

    fc1_1 = ll.DenseLayer(drop3_1, num_units=512, name='fc1_1')
    fc1_2 = ll.DenseLayer(drop3_2, num_units=512, W=fc1_1.W, b=fc1_1.b, name='fc1_2')

    drop3_1 = ll.DropoutLayer(fc1_1, p=0.5, name='drop3_1')
    drop3_2 = ll.DropoutLayer(fc1_2, p=0.5, name='drop3_2')


    fc2_1 = ll.DenseLayer(drop3_1, num_units=256, nonlinearity=linear, name='fc2_1')
    fc2_2 = ll.DenseLayer(drop3_2, num_units=256, nonlinearity=linear, W=fc2_1.W, b=fc2_1.b, name='fc2_2')

    desc_1 = ResponseNormalizationLayer(fc2_1, name='norm_1')
    desc_2 = ResponseNormalizationLayer(fc2_2, name='norm_2')

    # the incoming is of shape batch_size x 128, and we want it to be batch_size x 1 x 128 so we can concat and make the final output
    # batch_size x 2 x 128
    rs1_1 = ll.ReshapeLayer(desc_1, ([0], 1, [1]))
    rs1_2 = ll.ReshapeLayer(desc_2, ([0], 1, [1]))

    distance_out = ll.ConcatLayer([rs1_1, rs1_2], axis=1, name='concat_out') # 1 now the layer axis, 0 is batch axis, and 2 is feature axis

    # now let's try and put in a layer or two to predict if it's a match
    classifier_in = ll.ConcatLayer([desc_1, desc_2], axis=1, name='classifier_concat')

    classifier_dp1 = ll.DropoutLayer(classifier_in, p=0.5, name='classifier_dp1')
    classifier_fc1 = ll.DenseLayer(classifier_dp1, num_units=128, name='classifier_fc1')
    sigmoid_out = ll.DenseLayer(classifier_in, num_units=1, nonlinearity=sigmoid, name='classifier_out')


    # also return fc2_1 for the 'get_descriptor' function
    return distance_out, desc_1, sigmoid_out

def similarity_iter(output_layer, match_layer, update_params, match_layer_w=0):
    X1 = T.tensor4()
    X2 = T.tensor4()
    y = T.ivector()

    # find the input layers
    # TODO this better
    all_layers = ll.get_all_layers(match_layer)
    # make image of all layers
    imwrite_architecture(all_layers, './layer_rep.png')

    input_1 = filter(lambda x: x.name == 'input1', all_layers)[0]
    input_2 = filter(lambda x: x.name == 'input2', all_layers)[0]

    descriptors_train, match_prob_train = ll.get_output([output_layer, match_layer], {input_1: X1, input_2: X2})
    descriptors_eval, match_prob_eval = ll.get_output([output_layer, match_layer], {input_1: X1, input_2: X2}, deterministic=True)
    #descriptor_shape = ll.get_output_shape(output_layer, {input_1: X1, input_2: X2})
    #print("Network output shape: %r" % (descriptor_shape,))
    # distance minimization
    distance = lambda x: (x[:,0,:] - x[:,1,:] + 1e-7).norm(2, axis=1)
    #distance_eval = (descriptors_eval[:,0,:] - descriptors_eval[:,1,:] + 1e-7).norm(2, axis=1)
    # 9/21 squaring the loss seems to prevent it from getting to 0.5 really quickly (i.e. w/in 3 epochs)
    # let's see if it will learn something good
    margin = 1
    decay = 1e-3
    reg = regularize_network_params(match_layer, l2) * decay
    loss = lambda x, z: ((1-match_layer_w)*T.mean(y*(distance(x)) + (1 - y)*(T.maximum(0, margin - distance(x))))/2 # constrastive loss
            + match_layer_w*T.mean(binary_crossentropy(z.T + 1e-7,y))) # matching loss
    loss_reg = lambda x, z: (loss(x,z) + reg)
    # this loss doesn't work since it just pushes all the descriptors near each other and then predicts 0 all the time for tha matching
    #jason_loss = lambda x, z: T.mean(distance(x)*y + (1-y)*binary_crossentropy(z.T + 1e-7,y))
    #loss_eval = T.mean(y*(distance_eval**2) + (1 - y)*(T.maximum(0, 1 - distance_eval)**2))
    all_params = ll.get_all_params(match_layer) # unsure how I would do this if there were truly two trainable branches...
    loss_train = loss_reg(descriptors_train, match_prob_train)
    loss_train.name = 'combined_loss' # for the names
    grads = T.grad(loss_train, all_params, add_names=True)
    #updates = adam(grads, all_params, **update_params)
    updates = nesterov_momentum(grads, all_params, **update_params)

    train_iter = theano.function([X1, X2, y], [loss_train, loss(descriptors_train, match_prob_train)] + grads, updates=updates)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    valid_iter = theano.function([X1, X2, y], loss(descriptors_eval, match_prob_eval))

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def normalize_image(img):
    return (img - 128) / 255.

def img_batch_loader(imgdir, img_shape, names):
    return np.array([normalize_image(load_whole_image(imgdir, name, img_shape)) for name in names],dtype=np.float32).reshape(-1,3,*img_shape)

def dataset_prep(dataset):
    # dataset comes in as a list of pairs, split them up
    new_dataset = {}
    for split in dataset:
        new_dataset[split] = {'X1':[],'X2':[],'y':[]}
        for match_pair, is_match in zip(*dataset[split]):
            new_dataset[split]['X1'].append(match_pair[0])
            new_dataset[split]['X2'].append(match_pair[1])
            new_dataset[split]['y'].append(is_match)
        new_dataset[split]['y'] = np.array(new_dataset[split]['y'],dtype=np.int32)
    return new_dataset

def main(dataset_name, n_epochs):
    dataset = dataset_prep(load_dataset(join(join(dataset_loc,'Flukes/patches'),dataset_name)))
    img_dir = join(dataset_loc,'Flukes/CRC_combined constrained')
    batch_size = 32
    losses = {}
    epoch_losses = []
    batch_losses = []
    img_shape = (87,240)
    patch_layer, descriptor_layer, match_layer = build_siamese_whole(img_shape)
    iter_funcs = similarity_iter(patch_layer, match_layer, {'learning_rate':1e-2}, match_layer_w=0.5)
    batch_loader = partial(img_batch_loader, img_dir, img_shape)
    for epoch in range(n_epochs):
        tic = time.time()
        print("Epoch %d" % epoch)
        loss = train_epoch(iter_funcs, dataset, batch_size=batch_size, batch_loader=batch_loader)
        epoch_losses.append(loss['train_loss'])
        batch_losses.append(loss['all_train_loss'])
        # shuffle training set
        dataset['train'] = shuffle_dataset(dataset['train'])
        toc = time.time() - tic
        print("Train loss (reg): %0.3f\nTrain loss: %0.3f\nValid loss: %0.3f" %
                (loss['train_reg_loss'],loss['train_loss'],loss['valid_loss']))
        print("Took %0.2f seconds" % toc)
    batch_losses = list(chain(*batch_losses))
    losses['batch'] = batch_losses
    losses['epoch'] = epoch_losses
    parameter_analysis(match_layer)
    desc = desc_func(descriptor_layer)
    #display_losses(losses, n_epochs, batch_size, dataset['train']['X1'].shape[0])

    # Evaluate train rank accuracy and val rank accuracy
    #identifier_eval_dataset = load_identifier_eval(join(join(dataset_loc,'Flukes/patches'),dataset_name))
    #norm_idmap = normalize_image_idmap(identifier_eval_dataset['idmap'])
    #check_for_dupes(identifier_eval_dataset['idmap'])
    #print("Identification performance train:")
    #print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['train'], norm_idmap))
    #print("Identification performance valid:")
    #print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['val'], norm_idmap))

if __name__ == '__main__':
    try:
        dset_name = sys.argv[1]
        n_epochs = sys.argv[2]
    except IndexError:
        print("Usage: %s <dataset_name> <n_epochs>" % sys.argv[0])
        sys.exit(1)

    main(dset_name, int(n_epochs))
