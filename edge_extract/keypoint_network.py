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

def build_kpextractor64():
    inp = ll.InputLayer(shape=(None, 1, 64, 64), name='input')
    # we're going to build something like what Daniel Nouri made for Facial Keypoint detection for a base reference
    # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 32 x 32
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 16 x 16
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 8 x 8
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    # larger max pool to reduce parameters in the FC layer
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2) # now down to 4x4
    bn4 = ll.BatchNormLayer(mp4)
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2) # down to 2x2
    bn5 = ll.BatchNormLayer(mp5)
    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(bn5, num_units=256, nonlinearity=rectify)
    bn6 = ll.BatchNormLayer(fc1)
    #dp1 = ll.DropoutLayer(bn1, p=0.5)
    fc2 = ll.DenseLayer(bn6, num_units=64, nonlinearity=rectify)
    #dp2 = ll.DropoutLayer(fc2, p=0.5)
    bn7 = ll.BatchNormLayer(fc2)
    out = ll.DenseLayer(bn7, num_units=6, nonlinearity=linear)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs

def build_kpextractor128():
    inp = ll.InputLayer(shape=(None, 1, 128, 128), name='input')
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 64 x 64
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 32 x 32
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 16 x 16
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2) # now down to 8 x 8
    bn4 = ll.BatchNormLayer(mp4)
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2) # down to 4 x 4
    bn5 = ll.BatchNormLayer(mp5)

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2) # down to 4 x 4
    bn6 = ll.BatchNormLayer(mp6)
    dp0 = ll.DropoutLayer(bn6, p=0.5)

    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(bn6, num_units=256, nonlinearity=rectify)
    bn1_fc = ll.BatchNormLayer(fc1)
    dp1 = ll.DropoutLayer(bn1_fc, p=0.5)
    fc2 = ll.DenseLayer(dp1, num_units=64, nonlinearity=rectify)
    bn2_fc = ll.BatchNormLayer(fc2)
    dp2 = ll.DropoutLayer(bn2_fc, p=0.2)
    out = ll.DenseLayer(dp2, num_units=6, nonlinearity=linear)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs

def build_kpextractor128_decoupled():
    inp = ll.InputLayer(shape=(None, 1, 128, 128), name='input')
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2) # now down to 64 x 64
    bn1 = ll.BatchNormLayer(mp1)
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2) # now down to 32 x 32
    bn2 = ll.BatchNormLayer(mp2)
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2) # now down to 16 x 16
    bn3 = ll.BatchNormLayer(mp3)
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2) # now down to 8 x 8
    bn4 = ll.BatchNormLayer(mp4)
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2) # down to 4 x 4
    bn5 = ll.BatchNormLayer(mp5)

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2) # down to 4 x 4
    bn6 = ll.BatchNormLayer(mp6)
    dp0 = ll.DropoutLayer(bn6, p=0.5)

    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(dp0, num_units=256, nonlinearity=rectify)
    bn1_fc = ll.BatchNormLayer(fc1)
    dp1 = ll.DropoutLayer(bn1_fc, p=0.5)
    # so what we're going to do here instead is break this into three separate layers (each 32 units)
    # then each of these layers goes into a separate out, and out_rs will be a merge and then reshape
    fc2_left = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify)
    fc2_right = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify)
    fc2_notch = ll.DenseLayer(dp1, num_units=32, nonlinearity=rectify)

    out_left = ll.DenseLayer(fc2_left, num_units=2, nonlinearity=linear)
    out_right = ll.DenseLayer(fc2_right, num_units=2, nonlinearity=linear)
    out_notch = ll.DenseLayer(fc2_notch, num_units=2, nonlinearity=linear)

    out = ll.ConcatLayer([out_left, out_right, out_notch], axis=1)
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2))

    return out_rs

def build_kpextractor128_decoupled_stn():
    inp = ll.InputLayer(shape=(None, 1, 128, 128), name='input')
    identW, identb = make_identity_transform()
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    # Run an STN on the input w/a fairly complex localization network (ideally to identify where the Fluke is)
    # LOCALISATION NETWORK SECTION
    loc_conv1 = ll.Conv2DLayer(inp, num_filters=8, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='loc_conv1')
    loc_mp1 = ll.MaxPool2DLayer(loc_conv1, 2, stride=2, name='loc_mp1') # now down to 64 x 64
    loc_bn1 = ll.BatchNormLayer(loc_mp1, name='loc_bn1')


    loc_conv2 = ll.Conv2DLayer(loc_bn1, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='loc_conv2')
    loc_mp2 = ll.MaxPool2DLayer(loc_conv2, 2, stride=2, name='loc_mp2') # now down to 32 x 32
    loc_bn2 = ll.BatchNormLayer(loc_mp2, name='loc_bn2')

    loc_fc1 = ll.DenseLayer(loc_bn2, num_units=1024, nonlinearity=rectify, name='loc_fc1')
    loc_bn_fc1 = ll.BatchNormLayer(loc_fc1, name='loc_bn_fc1')
    loc_fc2 = ll.DenseLayer(loc_bn_fc1, num_units=256, nonlinearity=sigmoid, name='loc_fc2')
    loc_bn_fc2 = ll.BatchNormLayer(loc_fc2, name='loc_bn_fc2')

    loc_M = ll.DenseLayer(loc_bn_fc2, num_units=6, W=identW, b=identb, nonlinearity=linear, name='loc_M')
    stn = ll.TransformerLayer(inp, loc_M, name='stn')

    # MAIN NETWORK SECTION

    conv1 = ll.Conv2DLayer(stn, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2, name='mp1') # now down to 64 x 64
    bn1 = ll.BatchNormLayer(mp1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp2') # now down to 32 x 32
    bn2 = ll.BatchNormLayer(mp2, name='bn2')
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2, name='mp3') # now down to 16 x 16
    bn3 = ll.BatchNormLayer(mp3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp4') # now down to 8 x 8
    bn4 = ll.BatchNormLayer(mp4, name='bn4')
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2, name='mp5') # down to 4 x 4
    bn5 = ll.BatchNormLayer(mp5, name='bn5')

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp6') # down to 2 x 2
    bn6 = ll.BatchNormLayer(mp6, name='bn6')
    dp0 = ll.DropoutLayer(bn6, p=0.5, name='dp0')

    # now let's bring it down to a FC layer that takes in the 2x2x64 mp4 output
    fc1 = ll.DenseLayer(dp0, num_units=256, nonlinearity=rectify, name='fc1')
    bn1_fc = ll.BatchNormLayer(fc1, name='bn1_fc')
    dp1 = ll.DropoutLayer(bn1_fc, p=0.5, name='dp1')
    # since we don't want to batch normalize or dropout the transformation params, we'll concatenate them in afterwards
    dp1_and_loc_M = ll.ConcatLayer([dp1, loc_M], axis=1)
    # so what we're going to do here instead is break this into three separate layers (each 32 units)
    # then each of these layers goes into a separate out, and out_rs will be a merge and then reshape
    fc2_left = ll.DenseLayer(dp1_and_loc_M, num_units=32, nonlinearity=rectify, name='fc2_left')
    fc2_right = ll.DenseLayer(dp1_and_loc_M, num_units=32, nonlinearity=rectify, name='fc2_right')
    fc2_notch = ll.DenseLayer(dp1_and_loc_M, num_units=32, nonlinearity=rectify, name='fc2_notch')

    out_left = ll.DenseLayer(fc2_left, num_units=2, nonlinearity=linear, name='out_left')
    out_right = ll.DenseLayer(fc2_right, num_units=2, nonlinearity=linear, name='out_right')
    out_notch = ll.DenseLayer(fc2_notch, num_units=2, nonlinearity=linear, name='out_notch')

    out = ll.ConcatLayer([out_left, out_right, out_notch], axis=1, name='out')
    out_rs = ll.ReshapeLayer(out, ([0], 3, 2), name='out_rs')

    return out_rs



def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y

    imgs = np.expand_dims(dataset[0].astype(np.float32),axis=1) # add fake channel axis to greyscaled images
    #imgs_T = np.transpose(imgs,axes=(0,1,3,2)) # the most insane thing I've done
    imgs_T = imgs
    #print(imgs_T[0])

    img_shape = imgs_T.shape[-2:]
    # assume img_shape is y, x

    # scale the points to 0, 1
    # 'true' points are already rescaled to the given image shape
    pts = dataset[1].astype(np.float32)
    scale_mat = np.array([img_shape[1], img_shape[0]] * pts.shape[1], dtype=np.float32).reshape(pts.shape[1], 2)
    pts_scaled = (pts) / (scale_mat)
    #print(pts_scaled)

    # extract the original image sizes given that they're paired with the image names
    sizes = np.stack(dataset[2][:,0],axis=0).astype(np.float32)[:,:2]

    return shuffle_dataset({'X':imgs_T, 'y':pts_scaled, 'extra':sizes, 'names':list(dataset[2][:,1])})



def loss_iter(kpextractor, update_params={}):
    X = T.tensor4()
    y = T.tensor3() # first axis is batch axis, second is point axis (left, right, notch) third is x, y
    sizes = T.matrix() # first axis is batch, then y, x
    #itr = T.scalar()

    all_layers = ll.get_all_layers(kpextractor)
    imwrite_architecture(all_layers, './layer_rep_kpext.png')
    predicted_points_nondet = ll.get_output(kpextractor, X)
    predicted_points_det = ll.get_output(kpextractor, X, deterministic=True)

    # originally had another multiplicative factor: alpha, but this can be worked into the lr
    rmse = lambda pred, true: T.sqrt(T.mean((pred - true)**2, axis=(1,2)))
    eucl = lambda pred, true: T.mean((pred - true).norm(2, axis=2), axis=1)
    scaler = T.extra_ops.repeat(sizes[:,np.newaxis,:], 3, axis=1)[:,:,::-1]
    scaled_cost = lambda pred, alpha: eucl(pred*scaler*alpha, y*scaler*alpha)

    # experiment: penalize the difference between the avg std of the output over the batch
    # and the avg std of the true points over the batch
    std_diff = lambda pred: T.sqrt(T.mean((T.std(pred,axis=0) - T.std(y,axis=0))**2))
    losses = lambda pred: T.mean(scaled_cost(pred, 0.002))
    decay = 1e-5
    reg = regularize_network_params(kpextractor, l2) * decay

    #predT_p = theano.printing.Print()(T.mean(predicted_points_train,axis=0))
    losses_reg = lambda pred: losses(pred) + reg
    loss_train = losses_reg(predicted_points_nondet)
    loss_train.name = 'SE'
    all_params = ll.get_all_params(kpextractor, trainable=True)
    #all_params = filter(lambda x: not(x.name.startswith('loc')), all_params)
    grads = T.grad(loss_train, all_params, add_names=True)

    updates = nesterov_momentum(grads, all_params, update_params['l_r'], momentum=update_params['momentum'])
    #updates[update_params['l_r']] = update_params['l_r'] * 0.999
    #updates = adam(grads, all_params)

    # calculate and report average pixel distance in the real image
    #sizey, sizex = all_layers[0].shape[2:]
    #scale_denom = np.array([sizex, sizey]*3,dtype=np.float32).reshape(3,2)
    #scale_num = T.extra_ops.repeat(sizes[:,np.newaxis,:], 3, axis=1)[:,:,::-1]
    #scale_true = (scale_num / scale_denom)
    avg_pix_dist = lambda pred, alpha: T.mean((pred*scaler*alpha - y*scaler*alpha).norm(2, axis=2),axis=0)


    #pix_dist_train = avg_pix_dist(predicted_points_nondet, 1)
    pix_dist_det = avg_pix_dist(predicted_points_det, 1)

    print("Compiling network for training")
    tic = time.time()
    # use the valid function to determine actual training loss at the end of an epoch
    train_iter = theano.function([X, y, sizes], [loss_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X, y, sizes], [losses(predicted_points_det), losses_reg(predicted_points_det),
                                          pix_dist_det],
                                          #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False),
                                          )

    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def augmenting_batch_loader(dataset, bslice, sec):
    if sec != 'train':
        return [dataset[s][bslice] for s in ['X','y','extra']]
    else:
        Xaug, yaug = transform(dataset['X'][bslice], dataset['y'][bslice]*dataset['X'][bslice].shape[-1], transform_y=True)
        yaug /= dataset['X'][bslice].shape[-1]
        #with open(join(dataset_loc, "Flukes/kpts/sample_aug.pkl"), 'w') as f:
        #    pickle.dump((dataset['X'][bslice], dataset['y'][bslice], Xaug, yaug), f)
        return [Xaug, yaug, dataset['extra'][bslice]]

def nonaugmenting_batch_loader(dataset, bslice, sec):
    return [dataset[s][bslice] for s in ['X','y','extra']]

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
    dset = load_dataset(join(dataset_loc, "Flukes/kpts/%s" % dset_name), normalize_method='zscore')
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    epoch_losses = []
    batch_losses = []
    kp_extractor = build_kpextractor128_decoupled_stn()
    model_path = join(dataset_loc, "Flukes/kpts/%s/model.pkl" % dset_name)
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
