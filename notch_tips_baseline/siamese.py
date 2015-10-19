from __future__ import division
from itertools import chain, product
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
        train_epoch)

def build_siamese_separate_similarity(batch_size=32):
    # siamesse model similar to http://nbviewer.ipython.org/gist/ebenolson/40205d53a1a27ed0cc08#

    # Build first network

    inp_1 = ll.InputLayer(shape=(None, 3, 128, 128), name='input1')
    inp_2 = ll.InputLayer(shape=(None, 3, 128, 128), name='input2')


    # learning pose normalization might help maybe?
    identW, identb = make_identity_transform()
    loc1_1 = ll.DenseLayer(inp_1, num_units=6, W=identW, b=identb, nonlinearity=linear, name='loc1_1')
    loc1_2 = ll.DenseLayer(inp_2, num_units=6, W=loc1_1.W, b=loc1_1.b, nonlinearity=linear, name='loc1_2')

    st1_1 = ll.TransformerLayer(inp_1, loc1_1, name='st1_1')
    st1_2 = ll.TransformerLayer(inp_2, loc1_2, name='st1_2')

    # keeping it on 'same' for padding and a stride of 1 leaves the output of conv2D the same 2D shape as the input, which is convenient
    # Following Sander's guidelines here https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/cv3ib5q
    # conv2d defaults to Relu with glorot init
    conv1_1 = ll.Conv2DLayer(st1_1, num_filters=128, filter_size=(3,3), pad='same', W=Orthogonal(), name='conv1_1')
    conv1_2 = ll.Conv2DLayer(st1_2, num_filters=128, filter_size=(3,3), pad='same', W=conv1_1.W, b=conv1_1.b, name='conv1_2')

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

def dataset_prep(original_dataset):
    # load the original dataset, which contains pairs of patchsets and whether or not they match
    # we want to get this into the form of three separate datasets, for each of left, right, notch
    # each of these will consist of two 4-tensors, the first axis being the sample axis, and the rest being (channel, width, height)
    # the y values will be whether or not X1[i] matches X2[i]

    # then organize this structure into a dictionary for each of train valid and test for each of the three, and train three models
    tic = time.time()
    patch_dataset = {patch_type:{subset:{'X1':[], 'X2':[], 'y':[]}
                    for subset in ['train','valid','test']}
                    for patch_type in ['notch','left','right']}
    for subset in original_dataset:
        # train / val / test
        for match_pair, is_match in zip(*original_dataset[subset]):
            match_pair = (normalize_patches(match_pair[0]), normalize_patches(match_pair[1]))
            for patch_type in ['notch','left','right']:
                patch_dataset[patch_type][subset]['X1'].append(match_pair[0][patch_type])
                patch_dataset[patch_type][subset]['X2'].append(match_pair[1][patch_type])
                patch_dataset[patch_type][subset]['y'].append(is_match)
    for patch_type in patch_dataset:
        for subset in patch_dataset[patch_type]:
            patch_dataset[patch_type][subset]['X1'] = np.array(patch_dataset[patch_type][subset]['X1'],dtype=np.float32).swapaxes(1,3)
            patch_dataset[patch_type][subset]['X2'] = np.array(patch_dataset[patch_type][subset]['X2'],dtype=np.float32).swapaxes(1,3)
            patch_dataset[patch_type][subset]['y'] = np.array(patch_dataset[patch_type][subset]['y'], dtype=np.int32)
            #print(["%s : %s : %s" % (patch_type, subset, patch_dataset[patch_type][subset][sec].shape[0]) for sec in ['X1','X2','y']])
    return patch_dataset

def get_patches(individual_list, patchmap):
    # return a list of all the patches for each individual
    patches = {patch_type:{'id':[],'data':[]} for patch_type in ['notch','left','right']}
    for indv in individual_list:
        for patchset in patchmap[indv]:
            for patch_type in patchset:
                patches[patch_type]['id'].append(indv)
                patches[patch_type]['data'].append(patchset[patch_type])

    return patches

def compute_descriptors(descriptor_funcs, patches, batch_size=32):
    # assume patches is organized like it is in get_patches
    for patch_type in patches:
        nbatch = (len(patches[patch_type]['id']) // batch_size) + 1
        patches[patch_type]['desc'] = []
        for batch_ind in range(nbatch):
            batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
            batch = np.array(patches[patch_type]['data'][batch_slice],dtype='float32').reshape(-1,3,128,128)
            descriptors = descriptor_funcs[patch_type](batch)
            patches[patch_type]['desc'].append(descriptors)
        patches[patch_type]['desc'] = list(chain(*patches[patch_type]['desc']))
        # debug
        allzero_counter = 0
        anynan_counter = 0
        for desc in patches[patch_type]['desc']:
            if not any(desc):
                allzero_counter += 1
            if any(np.isnan(desc)):
                anynan_counter += 1
        if allzero_counter > 0:
            print("Patch type %s has %d all zero descriptors" % (patch_type, allzero_counter))
        if anynan_counter > 0:
            print("Patch type %s has %d descriptors with a nan" % (patch_type, anynan_counter))
        random_ind = random.randint(0,len(patches[patch_type]['desc']))
        random_ind2 = random.randint(0,len(patches[patch_type]['desc']))
        while (random_ind2 == random_ind):
            random_ind2 = random.randint(0,len(patches[patch_type]['desc']))
        #print("Two random random descriptors for patch %s: %r\n%r" % (patch_type,
        #    patches[patch_type]['desc'][random_ind],
        #    patches[patch_type]['desc'][random_ind2]))
        print("Distance btw two random desc %0.2f" % np.linalg.norm(patches[patch_type]['desc'][random_ind] - patches[patch_type]['desc'][random_ind2]))
        print("Do they match? %s" % (patches[patch_type]['id'][random_ind] == patches[patch_type]['id'][random_ind2]))
        #print("Random descriptor norm %0.2f" % np.linalg.norm(patches[patch_type]['desc'][random_ind],axis=0))
    return patches

def get_distances(query_desc_ind, patchset):
    # return a dict of all individuals mapped to average descriptor distance
    # this is returned for each patch type, and also the average across the patch types
    desc_distances = {'avg':{}}
    for patch_type in patchset:
        query_desc = patchset[patch_type]['desc'][query_desc_ind]
        ids = [id_ for ind, id_ in enumerate(patchset[patch_type]['id']) if ind != query_desc_ind]
        descs = [desc for ind, desc in enumerate(patchset[patch_type]['desc']) if ind != query_desc_ind]
        #ids = patchset[patch_type]['id']
        #descs = patchset[patch_type]['desc']
        distances = np.linalg.norm(descs - query_desc, axis=1)
        # compute the average distance for each individual
        avg_distances = {id_:[] for id_ in ids}
        for ind, id_ in enumerate(ids):
            avg_distances[id_].append(distances[ind])
        for id_ in avg_distances:
            avg_distances[id_] = np.min(avg_distances[id_])
        desc_distances[patch_type] = avg_distances
        for id_ in avg_distances:
            if id_ in desc_distances['avg']:
                desc_distances['avg'][id_].append(avg_distances[id_])
            else:
                desc_distances['avg'][id_] = [avg_distances[id_]]
    for id_ in desc_distances['avg']:
        desc_distances['avg'][id_] = np.average(desc_distances['avg'][id_])
    return desc_distances

def identifier_eval(descriptor_funcs, individual_list, patchmap, k=5, batch_size=32):
    tic = time.time()
    dataset = compute_descriptors(descriptor_funcs, get_patches(individual_list, patchmap), batch_size=batch_size)
    num_desc = len(dataset['notch']['id']) # ok this should be fixed
    cdfs = [{ptype:0 for ptype in ['avg','notch','left','right']} for _ in range(k+1)] # the last one being 'not found'
    for q in range(num_desc):
        true_id = dataset['notch']['id'][q]
        id_distances = get_distances(q, dataset)
        #print(id_distances.keys())
        closest_k = {ptype: sorted(id_distances[ptype].keys(),key=lambda x: id_distances[ptype][x])[:k] for ptype in id_distances}
        #print(closest_k.keys())
        for ptype in closest_k:
            """
            if ptype != 'avg':

                if closest_k[ptype][0] != true_id:
                    print("Wat %s" % ptype)
                    print(true_id)
                    print(closest_k[ptype][0])
                    print(id_distances[ptype][true_id])
                    print(id_distances[ptype][closest_k[ptype][0]])
                    sys.exit()
            """
            try:
                rank_loc = closest_k[ptype].index(true_id)
                cdfs[rank_loc][ptype] += 1
                """
                for _intopk in range(rank_loc, k):
                    cdfs[_intopk][ptype] += 1
                """

            except ValueError:
                # assume this means the correct id isn't in the top k
                cdfs[-1][ptype] += 1 # failure case

    for i in range(k+1):
        cdfs[i] = {p:cdfs[i][p] / num_desc for p in cdfs[i]}
        #print(cdfs[i].keys())
    toc = time.time() - tic
    print("Took %0.2f secconds" % toc)
    return cdfs

def print_cdfs(cdfs):
    print('\t' + '\t'.join(["%d" % (k+1) for k in range(len(cdfs)-1)]) + ('\t>%d' % (len(cdfs)-1)))
    for ptype in ['avg','notch','left','right']:
        print(ptype + '\t' + '\t'.join(["%0.3f" % cdf[ptype] for cdf in cdfs]))

def parameter_analysis(layer):
    all_params = ll.get_all_param_values(layer, regularizable=True)
    for param in all_params:
        print(param.shape)
        nneg_w = np.count_nonzero(param < 0) / np.product(param.shape)
        normed_norm = np.linalg.norm(param) / np.product(param.shape)
        print("Number of negative weights: %0.2f" % nneg_w)
        print("Weight norm (normalized by size): %0.10f" % normed_norm)

def display_losses(losses, n_epochs, batch_size, train_size, fn='losses.png'):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(3)
    batches_per_epoch = train_size // batch_size
    for ind, patch_type in enumerate(losses.keys()):
        axarr[ind].scatter(range(n_epochs*batches_per_epoch), losses[patch_type]['batch'], color='g')
        axarr[ind].scatter([i*batches_per_epoch for i in range(n_epochs)], losses[patch_type]['epoch'], color='r', s=10.)
        axarr[ind].set_title(patch_type)
    plt.savefig(fn)


def main(dataset_name, n_epochs):
    with open('../dataset_loc','r') as f:
        dataset_loc = f.read().rstrip()
    all_datasets = dataset_prep(load_dataset(join(join(dataset_loc,'Flukes/patches'),dataset_name)))
    batch_size = 32
    desc_funcs = {}
    losses = {}
    for patch_type in all_datasets:
        epoch_losses = []
        batch_losses = []
        patch_layer, descriptor_layer, match_layer = build_siamese_separate_similarity()
        iter_funcs = similarity_iter(patch_layer, match_layer, {'learning_rate':1e-2}, match_layer_w=0.5)
        for epoch in range(n_epochs):
            tic = time.time()
            print("%s: Epoch %d" % (patch_type, epoch))
            loss = train_epoch(iter_funcs, all_datasets[patch_type], batch_size=batch_size)
            epoch_losses.append(loss['train_loss'])
            batch_losses.append(loss['all_train_loss'])
            # shuffle training set
            #all_datasets[patch_type]['train'] = shuffle_dataset(all_datasets[patch_type]['train'])
            toc = time.time() - tic
            print("Train loss (reg): %0.3f\nTrain loss: %0.3f\nValid loss: %0.3f" %
                    (loss['train_reg_loss'],loss['train_loss'],loss['valid_loss']))
            print("Took %0.2f seconds" % toc)
        batch_losses = list(chain(*batch_losses))
        losses[patch_type] = {}
        losses[patch_type]['batch'] = batch_losses
        losses[patch_type]['epoch'] = epoch_losses
        print(patch_type)
        parameter_analysis(match_layer)
        desc_funcs[patch_type] = desc_func(descriptor_layer)
    display_losses(losses, n_epochs, batch_size, all_datasets['notch']['train']['X1'].shape[0])

    # Evaluate train rank accuracy and val rank accuracy
    identifier_eval_dataset = load_identifier_eval(join(join(dataset_loc,'Flukes/patches'),dataset_name))
    norm_idmap = normalize_image_idmap(identifier_eval_dataset['idmap'])
    #check_for_dupes(identifier_eval_dataset['idmap'])
    print("Identification performance train:")
    print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['train'], norm_idmap))
    print("Identification performance valid:")
    print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['val'], norm_idmap))

if __name__ == '__main__':
    try:
        dset_name = sys.argv[1]
        n_epochs = sys.argv[2]
    except IndexError:
        print("Usage: %s <dataset_name> <n_epochs>" % sys.argv[0])
        sys.exit(1)

    main(dset_name, int(n_epochs))
