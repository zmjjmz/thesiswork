from __future__ import division
from itertools import chain

import cPickle as pickle
from os.path import join
import time
from sklearn.utils import shuffle
import sys
import numpy as np
from ibeis_cnn.draw_net import imwrite_architecture

import lasagne.layers as ll
from lasagne.nonlinearities import linear
from lasagne.updates import adam
import theano.tensor as T
import theano

def build_siamese_separate_similarity(batch_size=32):
    # siamesse model similar to http://nbviewer.ipython.org/gist/ebenolson/40205d53a1a27ed0cc08#

    # Build first network

    inp_1 = ll.InputLayer(shape=(None, 3, 128, 128), name='input1')
    inp_2 = ll.InputLayer(shape=(None, 3, 128, 128), name='input2')

    # keeping it on 'same' for padding and a stride of 1 leaves the output of conv2D the same 2D shape as the input, which is convenient
    # Following Sander's guidelines here https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/cv3ib5q
    # conv2d defaults to Relu with glorot init
    conv1_1 = ll.Conv2DLayer(inp_1, num_filters=32, filter_size=(3,3), pad='same', name='conv1_1')
    conv1_2 = ll.Conv2DLayer(inp_2, num_filters=32, filter_size=(3,3), pad='same', W=conv1_1.W, b=conv1_1.b, name='conv1_2')

    mpool1_1 = ll.Pool2DLayer(conv1_1, pool_size=3, mode='max', name='mpool1_1')
    mpool1_2 = ll.Pool2DLayer(conv1_2, pool_size=3, mode='max', name='mpool1_2')

    drop1_1 = ll.DropoutLayer(mpool1_1, p=0.5, name='drop1_1')
    drop1_2 = ll.DropoutLayer(mpool1_2, p=0.5, name='drop1_2')
    # now we're down to 128 / 3 = 42x42 inputs

    # 9/20: reducing the amount of filters in conv2 from 64->16 to hopefully deal with some overfitting
    # No bueno, reducing to 8 didn't help either
    conv2_1 = ll.Conv2DLayer(drop1_1, num_filters=4, filter_size=(3,3), pad='same', name='conv2_1')
    conv2_2 = ll.Conv2DLayer(drop1_2, num_filters=4, filter_size=(3,3), pad='same', W=conv2_1.W, b=conv2_1.b, name='conv2_2')

    mpool2_1 = ll.Pool2DLayer(conv2_1, pool_size=3, mode='max', name='mpool2_1')
    mpool2_2 = ll.Pool2DLayer(conv2_2, pool_size=3, mode='max', name='mpool2_2')

    drop2_1 = ll.DropoutLayer(mpool2_1, p=0.5, name='drop2_1')
    drop2_2 = ll.DropoutLayer(mpool2_2, p=0.5, name='drop2_2')
    # now we're down to 42 / 3 = 14x14 inputs, and 64 of them so we'll have 14x14x16 = 3136 inputs

    # for similarity, we'll do a few FC layers before merging

    fc1_1 = ll.DenseLayer(drop2_1, num_units=256, name='fc1_1')
    fc1_2 = ll.DenseLayer(drop2_2, num_units=256, W=fc1_1.W, b=fc1_1.b, name='fc1_2')


    fc2_1 = ll.DenseLayer(fc1_1, num_units=128, nonlinearity=linear, name='fc2_1')
    fc2_2 = ll.DenseLayer(fc1_2, num_units=128, nonlinearity=linear, W=fc2_1.W, b=fc2_1.b, name='fc2_2')

    # the incoming is of shape batch_size x 128, and we want it to be batch_size x 1 x 128 so we can concat and make the final output
    # batch_size x 2 x 128
    rs1_1 = ll.ReshapeLayer(fc2_1, ([0], 1, [1]))
    rs1_2 = ll.ReshapeLayer(fc2_2, ([0], 1, [1]))

    distance_out = ll.ConcatLayer([rs1_1, rs1_2], axis=1, name='concat_out') # 1 now the layer axis, 0 is batch axis, and 2 is feature axis

    # so for now let's assume that comparing the length 512 descriptor works

    # also return fc2_1 for the 'get_descriptor' function
    return distance_out, fc2_1

def desc_func(desc_layer):
    X = T.tensor4()
    all_layers = ll.get_all_layers(desc_layer)
    imwrite_architecture(all_layers, './desc_function.png')

    descriptor = ll.get_output(desc_layer, X, deterministic=True)
    return theano.function([X], descriptor)


def similarity_iter(output_layer, update_params):
    X1 = T.tensor4()
    X2 = T.tensor4()
    y = T.ivector()

    # find the input layers
    # TODO this better
    all_layers = ll.get_all_layers(output_layer)
    # make image of all layers
    imwrite_architecture(all_layers, './layer_rep.png')

    input_1 = filter(lambda x: x.name == 'input1', all_layers)[0]
    input_2 = filter(lambda x: x.name == 'input2', all_layers)[0]

    descriptors_train = ll.get_output(output_layer, {input_1: X1, input_2: X2})
    descriptors_eval = ll.get_output(output_layer, {input_1: X1, input_2: X2}, deterministic=True)
    #descriptor_shape = ll.get_output_shape(output_layer, {input_1: X1, input_2: X2})
    #print("Network output shape: %r" % (descriptor_shape,))
    # distance minimization
    distance = lambda x: (x[:,0,:] - x[:,1,:] + 1e-7).norm(2, axis=1)
    #distance_eval = (descriptors_eval[:,0,:] - descriptors_eval[:,1,:] + 1e-7).norm(2, axis=1)
    # 9/21 squaring the loss seems to prevent it from getting to 0.5 really quickly (i.e. w/in 3 epochs)
    # let's see if it will learn something good
    loss = lambda x: T.mean(y*(distance(x)**2) + (1 - y)*(T.maximum(0, 1 - distance(x))**2))
    #loss_eval = T.mean(y*(distance_eval**2) + (1 - y)*(T.maximum(0, 1 - distance_eval)**2))
    all_params = ll.get_all_params(output_layer)
    updates = adam(loss(descriptors_train), all_params, **update_params)

    train_iter = theano.function([X1, X2, y], loss(descriptors_train), updates=updates)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    valid_iter = theano.function([X1, X2, y], loss(descriptors_eval))

    return {'train':train_iter, 'valid':valid_iter}

def train(iteration_funcs, dataset, batch_size=32):
    nbatch_train = (dataset['train']['y'].shape[0] // batch_size)
    nbatch_valid = (dataset['valid']['y'].shape[0] // batch_size)
    train_losses = []
    for batch_ind in range(nbatch_train):
        batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
        # this takes care of the updates as well
        batch_train_loss = iteration_funcs['train'](dataset['train']['X1'][batch_slice],
                                                    dataset['train']['X2'][batch_slice],
                                                    dataset['train']['y'][batch_slice])
        train_losses.append(batch_train_loss)

    avg_train_loss = np.mean(train_losses)

    valid_losses = []
    for batch_ind in range(nbatch_valid):
        batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
        # this takes care of the updates as well
        batch_valid_loss = iteration_funcs['valid'](dataset['valid']['X1'][batch_slice],
                                                    dataset['valid']['X2'][batch_slice],
                                                    dataset['valid']['y'][batch_slice])
        valid_losses.append(batch_valid_loss)

    avg_valid_loss = np.mean(valid_losses)

    return {'train_loss':avg_train_loss,'valid_loss':avg_valid_loss}


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
            for patch_type in ['notch','left','right']:
                patch_dataset[patch_type][subset]['X1'].append(match_pair[0][patch_type])
                patch_dataset[patch_type][subset]['X2'].append(match_pair[1][patch_type])
                patch_dataset[patch_type][subset]['y'].append(is_match)
    for patch_type in patch_dataset:
        for subset in patch_dataset[patch_type]:
            patch_dataset[patch_type][subset]['X1'] = np.array(patch_dataset[patch_type][subset]['X1']).reshape(-1,3,128,128)
            patch_dataset[patch_type][subset]['X2'] = np.array(patch_dataset[patch_type][subset]['X2']).reshape(-1,3,128,128)
            patch_dataset[patch_type][subset]['y'] = np.array(patch_dataset[patch_type][subset]['y'], dtype=np.int32)
            #print(["%s : %s : %s" % (patch_type, subset, patch_dataset[patch_type][subset][sec].shape[0]) for sec in ['X1','X2','y']])
    return patch_dataset


def load_dataset(dataset_path):
    print("Loading %s" % dataset_path)
    tic = time.time()
    dset = {}
    with open(join(dataset_path, 'train.pkl'),'r') as f:
        dset['train'] = pickle.load(f)
    with open(join(dataset_path, 'val.pkl'),'r') as f:
        dset['valid'] = pickle.load(f)
    with open(join(dataset_path, 'test.pkl'),'r') as f:
        dset['test'] = pickle.load(f)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    return dset

def load_identifier_eval(dataset_path):
    print("Loading %s for identification evaluation" % dataset_path)
    tic = time.time()
    dset = {}
    with open(join(dataset_path, 'idmap.pkl'), 'r') as f:
        dset['idmap'] = pickle.load(f)
    with open(join(dataset_path, 'train.indv'), 'r') as f:
        dset['train'] = pickle.load(f)
    with open(join(dataset_path, 'val.indv'), 'r') as f:
        dset['val'] = pickle.load(f)
    with open(join(dataset_path, 'test.indv'), 'r') as f:
        dset['test'] = pickle.load(f)
    return dset


def shuffle_dataset(dset):
    # assume dset has X1, X2, y
    X1, X2, y = shuffle(dset['X1'], dset['X2'], dset['y'])
    dset['X1'] = X1
    dset['X2'] = X2
    dset['y'] = y
    # TODO this more elegantly
    return dset

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
            batch = np.array(patches[patch_type]['data'][batch_slice]).reshape(-1,3,128,128)
            descriptors = descriptor_funcs[patch_type](batch)
            patches[patch_type]['desc'].append(descriptors)
        patches[patch_type]['desc'] = list(chain(*patches[patch_type]['desc']))
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
            avg_distances[id_] = np.average(avg_distances[id_])
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
    dataset = compute_descriptors(descriptor_funcs, get_patches(individual_list, patchmap), batch_size=batch_size)
    num_desc = len(dataset['notch']['id']) # ok this should be fixed
    cdfs = [{} for _ in range(k+1)] # the last one being 'not found'
    for q in range(num_desc):
        true_id = dataset['notch']['id'][q]
        id_distances = get_distances(q, dataset)
        closest_k = {ptype: sorted(id_distances[ptype].keys(),key=lambda x: id_distances[ptype][x])[:k] for ptype in id_distances}
        for ptype in id_distances:
            try:
                rank_loc = closest_k[ptype].index(true_id)
                for _intopk in range(rank_loc, k):
                    if ptype in cdfs[_intopk]:
                        cdfs[_intopk][ptype] += 1
                    else:
                        cdfs[_intopk][ptype] = 1

            except ValueError:
                # assume this means the correct id isn't in the top k
                if ptype in cdfs[-1]:
                    cdfs[-1][ptype] += 1 # failure case
                else:
                    cdfs[-1][ptype] = 1 # failure case

    for i in range(k+1):
        cdfs[i] = {p:cdfs[i][p] / num_desc for p in cdfs[i]}
    return cdfs

def print_cdfs(cdfs):
    for k, cdf in enumerate(cdfs):
        if k == len(cdfs) - 1:
            print("Not found in top-%d" % k)
        else:
            print("Found in top-%d"  % (k+1))
        for patch_type in cdf:
            print("%s : %0.2f" % (patch_type, cdf[patch_type]))


def main(dataset_name, n_epochs):
    with open('../dataset_loc','r') as f:
        dataset_loc = f.read().rstrip()
    original_dataset = load_dataset(join(join(dataset_loc,'Flukes/patches'),dataset_name))
    all_datasets = dataset_prep(original_dataset)
    desc_funcs = {}
    for patch_type in all_datasets:
        patch_layer, descriptor_layer = build_siamese_separate_similarity()
        iter_funcs = similarity_iter(patch_layer, {})
        for epoch in range(n_epochs):
            tic = time.time()
            print("%s: Epoch %d" % (patch_type, epoch))
            loss = train(iter_funcs, all_datasets[patch_type])
            # shuffle training set
            all_datasets[patch_type]['train'] = shuffle_dataset(all_datasets[patch_type]['train'])
            toc = time.time() - tic
            print(loss)
            print("Took %0.2f seconds" % toc)
        desc_funcs[patch_type] = desc_func(descriptor_layer)

    # Evaluate train rank accuracy and val rank accuracy
    identifier_eval_dataset = load_identifier_eval(join('/home/zj1992/windows/work2/datasets/Flukes/patches',dataset_name))
    print("Identification performance train:")
    print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['train'], identifier_eval_dataset['idmap']))
    print("Identification performance valid:")
    print_cdfs(identifier_eval(desc_funcs, identifier_eval_dataset['val'], identifier_eval_dataset['idmap']))

if __name__ == '__main__':
    try:
        dset_name = sys.argv[1]
        n_epochs = sys.argv[2]
    except IndexError:
        print("Usage: %s <dataset_name> <n_epochs>" % sys.argv[0])
        sys.exit(1)

    main(dset_name, int(n_epochs))
