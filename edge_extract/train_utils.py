from __future__ import division
from itertools import chain, product
import random
import cv2
from os.path import join, exists
from os import mkdir
import cPickle as pickle

from ibeis_cnn.draw_net import imwrite_architecture
import theano.tensor as T
from theano import function as tfunc
import lasagne.layers as ll
from lasagne.init import Constant
import numpy as np
import time
from sklearn.utils import shuffle
import scipy.sparse as ss

with open('../dataset_loc','r') as f:
    dataset_loc = f.read().rstrip()

class ResponseNormalizationLayer(ll.Layer):
    def get_output_for(self, input, **kwargs):
        return (input / (T.sqrt((input**2).sum(axis=1)).reshape((-1,1))))


def make_identity_transform():
    # returns b and W for identity transform
    b = np.zeros((2,3),dtype=np.float32)
    b[0,0] = 1
    b[1,1] = 1
    b = b.flatten()
    W = Constant(0.0)
    return W, b


def desc_func(desc_layer, save_diagram=True):
    # takes a layer and makes a function that returns its output
    # also saves a diagram of the network wrt the descriptor output
    X = T.tensor4()
    if save_diagram:
      all_layers = ll.get_all_layers(desc_layer)
      imwrite_architecture(all_layers, './desc_function.png')
    descriptor = ll.get_output(desc_layer, X, deterministic=True)
    return tfunc([X], descriptor)

def normalize_patch(patch, mean=128, std=255):
    return (patch - mean) / (std)


def normalize_patches(patchset):
    new_patchset = {}
    for patch in patchset:
        # zscore normalize
        #new_patchset[patch] = (patchset[patch] - np.mean(patchset[patch],axis=(0,1))) / np.std(patchset[patch],axis=(0,1))
        # assume that our dataset-wide avg is 128, and the std is 255
        new_patchset[patch] = normalize_patch(patchset[patch])
    return new_patchset

def normalize_image_pairs(img_pairlist):
    img_pairs = []
    for imgset1, imgset2 in img_pairlist:
        img_pairs.append((normalize_patches(imgset1),
                          normalize_patches(imgset2)))
    return img_pairs

def normalize_image_idmap(idmap):
    new_idmap = {}
    for indv in idmap:
        new_idmap[indv] = []
        for img in idmap[indv]:
            new_idmap[indv].append(normalize_patches(img))
    return new_idmap

def sparsify(data, threshold=0.6):
    if np.count_nonzero(data) / float(data.flatten().shape[0]) > threshold:
        original_shape = data.shape
        if len(data.shape) > 2:
            # ugh
            data = data.flatten()
        return (ss.coo_matrix(data), original_shape)
    else:
        return data

def desparsify(data):
    if isinstance(data, tuple):
        # this is definitely sparse in the very limited case that this code is meant to be used
        original_shape = data[1]
        datum = data[0]
        return datum.todense().reshape(original_shape)
    else:
        return data


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

def save_dataset(dataset_path, train, val, test):
    if exists(dataset_path):
        print("Overwriting %s y/n" % dataset_path)
        confirm = raw_input().rstrip()
        if confirm != 'y':
            return
    else:
        mkdir(dataset_path)
    tic = time.time()
    dset = {}
    with open(join(dataset_path, 'train.pkl'),'w') as f:
        pickle.dump(train,f)
    with open(join(dataset_path, 'val.pkl'),'w') as f:
        pickle.dump(val,f)
    with open(join(dataset_path, 'test.pkl'),'w') as f:
        pickle.dump(test,f)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)


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
    # assume dset has X, y, pixelw
    new_dset = {}
    X, y, pixelw = shuffle(dset['X'], dset['y'], dset['pixelw'])
    new_dset['X'] = X
    new_dset['y'] = y
    new_dset['pixelw'] = pixelw
    # TODO this more elegantly
    return new_dset

def check_for_dupes(idmap):
    # for every image, check if there's a duplicate image
    def is_equal(patchset1, patchset2):
        return all([not (np.any(patchset1[p] - patchset2[p])) for p in patchset1])
    for q_indv in idmap:
        # assume that no individuals have two images that are duplicates
        for d_indv in idmap:
            if d_indv == q_indv:
                continue
            for patchset1, patchset2 in product(idmap[q_indv], idmap[d_indv]):
                if is_equal(patchset1, patchset2):
                    print("%s has a duplicate image in %s's imageset" % (q_indv, d_indv))

def train_epoch(iteration_funcs, dataset, batch_size=32, batch_loader=(lambda x: x)):
    nbatch_train = (dataset['train']['y'].shape[0] // batch_size)
    nbatch_valid = (dataset['valid']['y'].shape[0] // batch_size)
    train_losses = []
    train_reg_losses = []

    train_accs = []
    valid_accs = []

    grad_mags = []
    grad_means = []
    for batch_ind in range(nbatch_train):
        batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
        # this takes care of the updates as well
        bloss_grads = iteration_funcs['train'](batch_loader(dataset['train']['X'][batch_slice]),
                                               dataset['train']['y'][batch_slice],
                                               dataset['train']['pixelw'][batch_slice])
        batch_train_loss_reg = bloss_grads.pop(0)
        batch_train_loss = bloss_grads.pop(0)
        batch_train_acc = bloss_grads.pop(0)
        grad_mags.append([np.linalg.norm(grad) for grad in bloss_grads])
        grad_means.append([np.mean(np.abs(grad)) for grad in bloss_grads])
        train_reg_losses.append(batch_train_loss_reg)
        train_losses.append(batch_train_loss)
        train_accs.append(batch_train_acc)

    avg_grad_mags = np.mean(np.array(grad_mags),axis=0)
    avg_grad_means = np.mean(np.array(grad_means),axis=0)
    print("Gradient names:\t%s" % iteration_funcs['gradnames'])
    print("Gradient magnitudes:\t%s" % avg_grad_mags)
    print("Gradient means:\t%s" % avg_grad_means)
    avg_train_loss = np.mean(train_losses)
    avg_train_reg_loss = np.mean(train_reg_losses)
    avg_train_acc = np.mean(train_accs)

    valid_losses = []
    for batch_ind in range(nbatch_valid):
        batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
        # this takes care of the updates as well
        batch_valid_loss, batch_valid_acc = iteration_funcs['valid'](batch_loader(dataset['valid']['X'][batch_slice]),
                                                    dataset['valid']['y'][batch_slice],
                                                    dataset['valid']['pixelw'][batch_slice])
        valid_losses.append(batch_valid_loss)
        valid_accs.append(batch_valid_acc)

    avg_valid_loss = np.mean(valid_losses)
    avg_valid_acc = np.mean(valid_accs)

    return {'train_loss':avg_train_loss,
            'train_reg_loss':avg_train_reg_loss,
            'valid_loss':avg_valid_loss,
            'train_acc':avg_train_acc,
            'valid_acc':avg_valid_acc,
            'all_train_loss':train_losses}

def load_whole_image(imgs_dir, img, img_shape=None):
    # imgs_dir should be the absolute path
    # loads the image and naively resizes to img_shape
    if img_shape is not None:
        return cv2.resize(cv2.imread(join(imgs_dir, img)), img_shape[::-1])
    else:
        return cv2.imread(join(imgs_dir, img))

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
    ax = plt.subplot()
    batches_per_epoch = train_size // batch_size
    ax.scatter(range(n_epochs*batches_per_epoch), losses['batch'], color='g')
    ax.scatter([i*batches_per_epoch for i in range(n_epochs)], losses['epoch'], color='r', s=10.)
    plt.savefig(fn)


