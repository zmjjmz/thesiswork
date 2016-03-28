from __future__ import division

from itertools import chain, product, permutations, combinations
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
from collections import defaultdict
import math
from sklearn.utils import resample, shuffle

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

def build_embedder():
    # assume input is a 256x256 image, bring it down to a 300-d (arbitrary) embedding
    inp = ll.InputLayer(shape=(None, 1, 256, 256), name='input')
    # alternate pooling and conv layers to minimize parameters
    filter_pad = lambda x, y: (x//2, y//2)
    filter3 = (3, 3)
    same_pad3 = filter_pad(*filter3)
    conv1 = ll.Conv2DLayer(inp, num_filters=16, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv1')
    mp1 = ll.MaxPool2DLayer(conv1, 2, stride=2, name='mp1') # now down to 128 x 128
    bn1 = ll.BatchNormLayer(mp1, name='bn1')
    conv2 = ll.Conv2DLayer(bn1, num_filters=32, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv2')
    mp2 = ll.MaxPool2DLayer(conv2, 2, stride=2, name='mp2') # now down to 64 x 64
    bn2 = ll.BatchNormLayer(mp2, name='bn2')
    conv3 = ll.Conv2DLayer(bn2, num_filters=64, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv3')
    mp3 = ll.MaxPool2DLayer(conv3, 2, stride=2, name='mp3') # now down to 32 x 32
    bn3 = ll.BatchNormLayer(mp3, name='bn3')
    conv4 = ll.Conv2DLayer(bn3, num_filters=128, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv4')
    mp4 = ll.MaxPool2DLayer(conv4, 2, stride=2, name='mp4') # now down to 16 x 16
    bn4 = ll.BatchNormLayer(mp4, name='bn4')
    conv5 = ll.Conv2DLayer(bn4, num_filters=256, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv5')
    mp5 = ll.MaxPool2DLayer(conv5, 2, stride=2, name='mp5') # down to 8 x 8
    bn5 = ll.BatchNormLayer(mp5, name='bn5')

    conv6 = ll.Conv2DLayer(bn5, num_filters=512, filter_size=filter3, pad=same_pad3, W=Orthogonal(), nonlinearity=rectify, name='conv6')
    mp6 = ll.MaxPool2DLayer(conv6, 2, stride=2, name='mp6') # down to 4 x 4 x 512 = 8192 units
    bn6 = ll.BatchNormLayer(mp6, name='bn6')
    dp0 = ll.DropoutLayer(bn6, p=0.5, name='dp0')
    # now let's do a few FC layers

    fc1 = ll.DenseLayer(dp0, num_units=4096, nonlinearity=rectify, name='fc1')
    bn_fc1 = ll.BatchNormLayer(fc1, name='bn_fc1')
    dp1 = ll.DropoutLayer(bn_fc1, p=0.5, name='dp1')
    fc2 = ll.DenseLayer(dp1, num_units=1024, nonlinearity=rectify, name='fc2')
    bn_fc2 = ll.BatchNormLayer(fc2, name='bn_fc2')
    dp2 = ll.DropoutLayer(bn_fc2, p=0.5, name='dp2')
    fc_embed = ll.DenseLayer(dp2, num_units=300, nonlinearity=linear, name='fc_embed')
    embed_norm = ResponseNormalizationLayer(fc_embed, name='embed_norm')

    return [embed_norm]

def preproc_dataset(dataset):
    # assume dataset is a tuple of X, y
    # split into singletons and multitons (fuckit that's a word)
    imgs = np.expand_dims(np.stack(dataset[0],axis=0).astype(np.float32),axis=1)
    ids = dataset[1]
    # preprocess positive_pairs
    # make an id_img_map mapping actual ids to indices into Xall
    id_img_map = defaultdict(lambda: [])
    for ind, id_ in enumerate(dataset[1]):
        id_img_map[id_].append(ind)
    positive_pairs = {'anchor':[], 'positive':[], 'id':[]}
    for id_ in id_img_map:
        if len(id_img_map[id_]) == 1:
            continue
        id_pairs = [{'anchor':anchor, 'positive':pos, 'id':id_} for anchor, pos in permutations(id_img_map[id_], 2)]
        for k in positive_pairs.keys():
            positive_pairs[k] = chain(positive_pairs[k], [id_pair[k] for id_pair in id_pairs])

    for k in positive_pairs.keys():
        positive_pairs[k] = list(positive_pairs[k])
    # preprocess negative_cands
    negative_cands = {}
    all_inds = range(imgs.shape[0])
    for id_ in id_img_map:
        # filter out the indices for the given id
        negative_cands[id_] = np.array(list(filter(lambda x: x not in id_img_map[id_], all_inds)))

    # build Xmul, a list of indices into Xall of multiton images
    # this is what's 'trained' on, i.e. where minibatches are pulled from
    # the main concern ofc is that the minibatches won't necessarily be the same size as
    # some anchors will go completely inactive at a given epoch, but we can't take them out of Xmul...
    # I think the best way to do this is actually not to use Xmul, but instead work through the keys
    # of active_dict

    all_pairs = list(combinations(range(imgs.shape[0]), 2))
    neg_pairs = list(filter(lambda inds: ids[inds[0]] != ids[inds[1]], all_pairs))
    pos_pairs = list(filter(lambda inds: ids[inds[0]] == ids[inds[1]], all_pairs))
    # sample neg_pairs (which is necessarily larger) down to the size of pos_pairs
    n_pos = len(pos_pairs)
    print(n_pos)
    neg_pairs_redux = resample(neg_pairs, n_samples=n_pos)
    sampled_pairs = shuffle(list(chain(pos_pairs, neg_pairs_redux)))

    all_pair_matches = np.array([int(ids[ind1] == ids[ind2]) for ind1, ind2 in sampled_pairs], dtype=np.int32)
    print(map(lambda x: ids[x], sampled_pairs[np.where(all_pair_matches)[0][0]]))

    return {'Xall':imgs, 'positive_pairs':positive_pairs, 'negative_cands':negative_cands, 'ids':ids,
            'pairs':sampled_pairs, 'y':all_pair_matches}

def contrastive_loss_iter(embedder, update_params={}):
    X_pairs = {
            'img1':T.tensor4(),
            'img2':T.tensor4(),
            }
    y = T.ivector() # basically class labels

    final_emb_layer = embedder[-1]
    all_layers = ll.get_all_layers(embedder)
    imwrite_architecture(all_layers, './layer_rep.png')
    # assume we get a list of predictions (e.g. for jet architecture, but should work w/just one pred)
    # another assumption (which must hold when the network is being made)
    # the last prediction layer is a) the end of the network and b) what we ultimately care about
    # however the other prediction layers will be incorporated into the training loss
    predicted_embeds_train = {k:ll.get_output(embedder, X)[-1] for k, X in X_pairs.items()}
    predicted_embeds_valid = {k:ll.get_output(final_emb_layer, X, deterministic=True) for k, X in X_pairs.items()}

    margin = 1

    # if distance is 0 that's bad
    distance = lambda pred: (pred['img1'] - pred['img2'] + 1e-7).norm(2, axis=1)
    contrastive_loss = lambda pred: T.mean(y*(distance(pred)) + (1 - y)*(margin - distance(pred)).clip(0,np.inf))
    failed_matches = lambda pred: T.switch(T.eq(T.sum(y),0), 0, T.sum((y*distance(pred)) > margin) / T.sum(y))
    failed_nonmatches = lambda pred: T.switch(T.eq(T.sum(1-y),0), 0, T.sum((1-y*distance(pred)) < margin) / T.sum(1-y))
    failed_pairs = lambda pred: 0.5*failed_matches(pred) + 0.5*failed_nonmatches(pred)

    decay = 0.0001
    reg = regularize_network_params(final_emb_layer, l2) * decay
    losses_reg = lambda pred: contrastive_loss(pred) + reg
    loss_train = losses_reg(predicted_embeds_train)
    loss_train.name = 'CL' # for the names
    #all_params = list(chain(*[ll.get_all_params(pred) for pred in embedder]))
    all_params = ll.get_all_params(embedder, trainable=True) # this should work with multiple 'roots'
    grads = T.grad(loss_train, all_params, add_names=True)
    updates = adam(grads, all_params)
    #updates = nesterov_momentum(grads, all_params, update_params['l_r'], momentum=update_params['momentum'])

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X_pairs['img1'], X_pairs['img2'], y], [loss_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X_pairs['img1'], X_pairs['img2'], y], [
                                    contrastive_loss(predicted_embeds_valid),
                                    losses_reg(predicted_embeds_valid),
                                    failed_pairs(predicted_embeds_valid)])
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}

def triplet_loss_iter(embedder, update_params={}):
    X_triplets = {
            'anchor':T.tensor4(),
            'positive':T.tensor4(),
            'negative':T.tensor4(),
            } # each will be a batch of images

    final_emb_layer = embedder[-1]
    all_layers = ll.get_all_layers(embedder)
    imwrite_architecture(all_layers, './layer_rep.png')
    # assume we get a list of predictions (e.g. for jet architecture, but should work w/just one pred)
    # another assumption (which must hold when the network is being made)
    # the last prediction layer is a) the end of the network and b) what we ultimately care about
    # however the other prediction layers will be incorporated into the training loss
    predicted_embeds_train = {k:ll.get_output(embedder, X)[-1] for k, X in X_triplets.items()}
    predicted_embeds_valid = {k:ll.get_output(final_emb_layer, X, deterministic=True) for k, X in X_triplets.items()}

    # each output should be batch_size x embed_size

    # should give us a vector of batch_size of distances btw anchor and positive
    alpha = 0.2 # FaceNet alpha
    triplet_pos = lambda pred: (pred['anchor'] - pred['positive']).norm(2,axis=1)
    triplet_neg = lambda pred: (pred['anchor'] - pred['negative']).norm(2,axis=1)
    triplet_distances = lambda pred: (triplet_pos(pred) - triplet_neg(pred) + alpha).clip(0, np.inf)
    triplet_failed = lambda pred: T.mean(triplet_distances(pred) > alpha)
    triplet_loss = lambda pred: T.sum(triplet_distances(pred))

    decay = 0.001
    reg = regularize_network_params(final_emb_layer, l2) * decay
    losses_reg = lambda pred: triplet_loss(pred) + reg
    loss_train = losses_reg(predicted_embeds_train)
    loss_train.name = 'TL' # for the names
    #all_params = list(chain(*[ll.get_all_params(pred) for pred in embedder]))
    all_params = ll.get_all_params(embedder, trainable=True) # this should work with multiple 'roots'
    grads = T.grad(loss_train, all_params, add_names=True)
    updates = adam(grads, all_params)
    #updates = nesterov_momentum(grads, all_params, update_params['l_r'], momentum=update_params['momentum'])

    print("Compiling network for training")
    tic = time.time()
    train_iter = theano.function([X_triplets['anchor'], X_triplets['positive'], X_triplets['negative']], [loss_train] + grads, updates=updates)
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    #theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    print("Compiling network for validation")
    tic = time.time()
    valid_iter = theano.function([X_triplets['anchor'], X_triplets['positive'], X_triplets['negative']], [triplet_loss(predicted_embeds_valid),
                                                                                                          losses_reg(predicted_embeds_valid),
                                                                                                          triplet_failed(predicted_embeds_valid)])
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)

    return {'train':train_iter, 'valid':valid_iter, 'gradnames':[g.name for g in grads]}


def make_batch(anchors, bslice, dataset, actives):
    # go through each image in the bslice, and pull triplets from the associated individuals
    # 'Xmul' will only have images associated w/multitons
    batches = {'anchor':[], 'positive':[], 'negative':[]}
    for image_ind, anchor in enumerate(anchors['y'][bslice]): # ugh
        if len(actives[anchor].keys()) == 0:
            continue
        # choose a random positive from the available positives for this anchor (will start very large)
        positive = random.choice(actives[anchor].keys()) # this is a list of positive image ids
        negative = random.choice(actives[anchor][positive]) # now a list of negative image ids

        # a test
        assert(dataset['ids'][anchor] == dataset['ids'][positive])
        assert(dataset['ids'][anchor] != dataset['ids'][negative])

        # construct the batch item
        batches['anchor'].append(dataset['Xall'][anchor])
        batches['positive'].append(dataset['Xall'][positive])
        batches['negative'].append(dataset['Xall'][negative]) # we want the singleton negatives as well
        batches = shuffle_dataset(batches)

        #if bslice.start == 0:
        #    ut.save_cPkl(join(dataset_loc, 'Flukes/patches/triplet_batches.pkl'), batches)

    return batches

def create_network_fn(embed_layer):
    # just take the embedder output, and make a tfn giving it some tensor4
    X = T.tensor4()
    #embedder = embed_builder()[-1]
    embed_out = ll.get_output(embed_layer, X, deterministic=True)
    embed_fn = theano.function([X], embed_out)

    return embed_fn

def batch_compute(lis, network_fn, batch_size):
    # just batches computation
    # assume lis is a np array w/batch axis as first axis
    nbatches = int(math.ceil(lis.shape[0] / batch_size))
    to_stack = []
    for batch_ind in range(nbatches):
        batch_slice = slice(batch_ind*batch_size, (batch_ind + 1)*batch_size)
        processed = network_fn(lis[batch_slice])
        to_stack.append(processed)

    try:
        stacked = np.concatenate(to_stack,axis=0)
    except ValueError:
        print(to_stack[0])
        raise

    return stacked

def find_actives(dataset, embedder_fn, batch_size=128, alpha=0.2, previous_actives=None):
    # so this is a bit nuts, but basically we're going to create a massive dictionary of dictionaries,
    # so first we'll add every multiton id, and do a n choose 2 thing over it, so each id has nC2 anchor/pos pairs
    # preproc_dataset should give us a map of multiton ids to indices in the dataset
    # NOTE: actually it doesn't make sense to recompute this part every time, we just need to ensure that if a given
    # positive pair is found w/no negatives that fail, it is removed
    positive_pairs = dataset['positive_pairs']
    # positive pairs consists of a dict: pos, anchor, id
    # assume they are indices into the main dataset (i.e. Xall)
    anchor_embeds = batch_compute(dataset['Xall'][positive_pairs['anchor']], embedder_fn, batch_size)
    pos_embeds = batch_compute(dataset['Xall'][positive_pairs['positive']], embedder_fn, batch_size)
    pos_dists = np.linalg.norm(anchor_embeds - pos_embeds, axis=1)

    # the negative candidates for every pair is going to be precomputed as well, as it won't change
    negative_cands = dataset['negative_cands']
    # this is a dict of id to list of indices into Xall suitable as negatives
    # probably faster to precompute, hopefully doesn't take too much memory...
    all_embeds = batch_compute(dataset['Xall'], embedder_fn, batch_size)
    assert(anchor_embeds.shape[0] == len(positive_pairs['id']))
    keep_negs = []
    for anch, pos_dist, id_ in zip(anchor_embeds, pos_dists, positive_pairs['id']): # this should be ok...
        neg_embeds = all_embeds[negative_cands[id_]]
        neg_dists = np.linalg.norm(anch - neg_embeds,axis=1)
        active_negs = negative_cands[id_][np.where(neg_dists < (pos_dist + alpha))]
        keep_negs.append(active_negs)

    # so now we'll go through each pair (i.e. zip positive_pairs anchor and positive)
    # and for each one, we'll construct the dictionary w/the associated keep_negs

    active_dict = defaultdict(lambda: {})
    pairs_solved = 0
    old_actives = []
    new_actives = []
    beaten_actives = []
    for anchor, positive, negs in zip(positive_pairs['anchor'], positive_pairs['positive'], keep_negs):
        if previous_actives is not None and anchor in previous_actives and positive in previous_actives[anchor]:
            prev_negs = previous_actives[anchor][positive]
            # determine the intersection of the previous negs and the new ones
            n_old_actives = len(set(negs).intersection(prev_negs))
            old_actives.append(n_old_actives)
            new_actives.append(len(negs) - n_old_actives)
            beaten_actives.append(len(prev_negs) - n_old_actives)
        if negs.shape[0] == 0:
            pairs_solved += 1
            continue #yay you won
        active_dict[anchor][positive] = negs
    print("Positive pairs beaten: %d / %d" % (pairs_solved, len(positive_pairs['anchor'])))
    if previous_actives is not None:
        print("Average number of old actives left over: %0.2f" % np.average(old_actives))
        print("Average number of new actives: %0.2f" % np.average(new_actives))
        print("Average number of beaten actives: %0.2f" % np.average(beaten_actives))
    return dict(active_dict)
    # O(m*N)

def full_triplets(active_triplets, sample_negs=10):
    # just get all the triplets
    triplets = {'anchor':[], 'positive':[], 'negative':[]}
    for anchor in active_triplets:
        for positive in active_triplets[anchor]:
            negs_shuffled = np.random.permutation(active_triplets[anchor][positive])
            negs_sample = negs_shuffled[:sample_negs]
            for negative in negs_sample:
                triplets['anchor'].append(anchor)
                triplets['positive'].append(positive)
                triplets['negative'].append(negative)
    triplets['y'] = np.array(range(len(triplets['anchor'])))
    triplets = shuffle_dataset(triplets)
    return triplets


def make_batch_full(triplets, bslice, dataset):
    batches = {'anchor':dataset['Xall'][triplets['anchor'][bslice]],
               'positive':dataset['Xall'][triplets['positive'][bslice]],
                'negative':dataset['Xall'][triplets['negative'][bslice]]
    }
    return batches

def make_batch_pairs(train_pairs, bslice, dataset):
    pairs1, pairs2 = map(list, zip(*train_pairs['pairs'][bslice]))
    batches = {'img1':dataset['Xall'][pairs1],
               'img2':dataset['Xall'][pairs2],
               'y':train_pairs['y'][bslice],
               'inds':train_pairs['pairs'][bslice]} # to manually verify that the ids are good
    if bslice.start == 0:
        ut.save_cPkl(join(dataset_loc, 'Flukes/patches/pair_batches.pkl'), batches)
    return batches



def augmenting_batch_loader(dataset, bslice, sec, batch_maker, keys):
    batch = batch_maker[sec](dataset, bslice)
    if sec != 'train':
        return [batch[s] for s in keys]
    else:
        return [transform(batch[s]) for s in keys]

def nonaugmenting_batch_loader(dataset, bslice, sec, batch_maker, keys):
    batch = batch_maker[sec](dataset, bslice)
    return [batch[s] for s in keys]

augmenting_batch_loader_trip = partial(augmenting_batch_loader, keys=['anchor','positive','negative'])
nonaugmenting_batch_loader_trip = partial(nonaugmenting_batch_loader, keys=['anchor','positive','negative'])

augmenting_batch_loader_pair = partial(augmenting_batch_loader, keys=['img1', 'img2', 'y'])
nonaugmenting_batch_loader_pair = partial(nonaugmenting_batch_loader, keys=['img1', 'img2', 'y'])


FUNCTIONS = {
    'triplet':{
        'loss_iter':triplet_loss_iter,
        'aug_bl':augmenting_batch_loader_trip,
        'nonaug_bl':nonaugmenting_batch_loader_trip,
        'bm':make_batch_full,
        },
    'pair':{
        'loss_iter':contrastive_loss_iter,
        'aug_bl':augmenting_batch_loader_pair,
        'nonaug_bl':nonaugmenting_batch_loader_pair,
        'bm':make_batch_pairs,
        },
}

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--test", action='store_true', dest='test')
    parser.add_option("-r", "--resume", action='store_true', dest='resume')
    parser.add_option("-d", "--dataset", action='store', type='string', dest='dataset')
    parser.add_option("-b", "--batch_size", action="store", type="int", dest='batch_size', default=32)
    parser.add_option("-e", "--epochs", action="store", type="int", dest="n_epochs", default=1)
    parser.add_option("-l", "--loss", action="store", type="string", dest="loss_type", default='triplet')
    options, args = parser.parse_args()
    if options.test:
        print("No test stuff defined")
        sys.exit(0)
    dset_name = options.dataset
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    if options.loss_type not in ('triplet', 'pair'):
        print("Invalid loss type: %s" % options.loss_type)
        sys.exit(1)
    print("Loading dataset")
    dset = load_dataset(join(dataset_loc, "Flukes/patches/%s" % dset_name), normalize_method='zscore')
    print("Figuring out triplet candidates and pairs")
    tic = time.time()
    dset = {section:preproc_dataset(dset[section]) for section in ['train', 'valid', 'test']}
    # load_dataset normalizes
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    epoch_losses = []
    batch_losses = []
    embedder = build_embedder()
    model_path = join(dataset_loc, "Flukes/patches/%s/model%s.pkl" % (dset_name, options.loss_type))
    if options.resume and exists(model_path):
        params = ut.load_cPkl(model_path)
        ll.set_all_param_values(embedder, params)

    print("Compiling network for embedding")
    simple_embedder_fn = create_network_fn(embedder[-1])
    #iter_funcs = loss_iter(embedder, update_params={'learning_rate':.01})
    lr = theano.shared(np.array(0.010, dtype=np.float32))
    momentum_params = {'l_r':lr, 'momentum':0.9}

    iter_funcs = FUNCTIONS[options.loss_type]['loss_iter'](embedder, update_params=momentum_params)
    best_params = ll.get_all_param_values(embedder)
    best_val_loss = np.inf
    layer_names = [p.name for p in ll.get_all_params(embedder, trainable=True)]
    save_model = True
    active_triplets = {'train':None, 'valid':None}
    try:
        for epoch in range(n_epochs):
            tic = time.time()
            print("Epoch %d" % (epoch))
            if options.loss_type == 'triplet':
                print("Finding active triplets")
                active_tic = time.time()
                active_triplets['train'] = find_actives(dset['train'], simple_embedder_fn, previous_actives=active_triplets['train'])
                active_triplets['valid'] = find_actives(dset['valid'], simple_embedder_fn, previous_actives=active_triplets['valid'])
                active_toc = time.time() - active_tic
                print("Took %0.2f seconds" % active_toc)
                # give the list of active anchors (i.e. the keys in active_triplets) as the dataset
                # 'y' key is a hack
                full_triplet_sets = {k:full_triplets(active_triplets[k]) for k in active_triplets}
                dataset = full_triplet_sets


                #anchor_sets = {k:{'y':np.random.permutation(np.array(active_triplets[k].keys()))} for k in active_triplets}

                print("Active anchors left: %r" % ({k:len(v) for k, v in active_triplets.items()}))
                print("Active triplets left: %r" % ({k:v['y'].shape[0] for k, v in full_triplet_sets.items()}))
                # build the batch loader as a partial function on the dataset and the actives
                #batch_maker = {k:partial(make_batch, dataset=dset[k], actives=active_triplets[k]) for k in ['train','valid']}
            else:
                print("Number of pairs: %r" % ({k:len(dset[k]['y']) for k in dset}))
                train_pairs_shuffled = {k:shuffle_dataset({part:dset[k][part] for part in ['pairs','y']})
                                        for k in ['train','valid']}
                assert(all(train_pairs_shuffled['train']['y'] ==
                    map(lambda x: int(dset['train']['ids'][x[0]] == dset['train']['ids'][x[1]]),
                                    train_pairs_shuffled['train']['pairs'])))
                dataset = train_pairs_shuffled


            batch_maker = {k:partial(FUNCTIONS[options.loss_type]['bm'], dataset=dset[k]) for k in ['train','valid']}
            batch_loader = partial(FUNCTIONS[options.loss_type]['nonaug_bl'], batch_maker=batch_maker)

            # so we're going to just give dset as
            loss = train_epoch(iter_funcs, dataset, batch_size, batch_loader, layer_names=layer_names)
            epoch_losses.append(loss['train_loss'])
            batch_losses.append(loss['all_train_loss'])
            toc = time.time() - tic
            print("Learning rate: %0.5f" % momentum_params['l_r'].get_value())
            print("Train loss (reg): %0.3f\nTrain loss: %0.3f\nValid loss: %0.3f" %
                    (loss['train_reg_loss'],loss['train_loss'],loss['valid_loss']))
            print("Train %s failed: %s\nValid %s failed: %s" % (options.loss_type, loss['train_acc'],
                                                                options.loss_type, loss['valid_acc'],))
            if loss['valid_loss'] < best_val_loss:
                best_params = ll.get_all_param_values(embedder)
                best_val_loss = loss['valid_loss']
                print("New best validation loss!")
            print("Took %0.2f seconds" % toc)
    except KeyboardInterrupt:
        print("Training interrupted, save model? y/n")
        confirm = raw_input().rstrip()
        if confirm == 'n':
            save_model = False

    batch_losses = list(chain(*batch_losses))
    losses = {}
    losses['batch'] = batch_losses
    losses['epoch'] = epoch_losses
    parameter_analysis(embedder)
    #display_losses(losses, n_epochs, batch_size, dset['train']['X'].shape[0])

    # TODO: move to train_utils and add way to load up previous model
    if save_model:
        ut.save_cPkl(model_path, best_params)
