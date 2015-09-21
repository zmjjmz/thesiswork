import cPickle as pickle
from os.path import join
import time
from sklearn.utils import shuffle
import sys
import numpy as np

import lasagne.layers as ll
from lasagne.nonlinearities import linear
from lasagne.updates import adam
import theano.tensor as T
import theano

def build_siamese_separate_similarity():
    # siamesse model similar to http://nbviewer.ipython.org/gist/ebenolson/40205d53a1a27ed0cc08#

    # Build first network

    inp_1 = ll.InputLayer(shape=(None, 3, 128, 128), name='input1')
    inp_2 = ll.InputLayer(shape=(None, 3, 128, 128), name='input2')

    # keeping it on 'same' for padding and a stride of 1 leaves the output of conv2D the same 2D shape as the input, which is convenient
    # Following Sander's guidelines here https://www.reddit.com/r/MachineLearning/comments/3l5qu7/rules_of_thumb_for_cnn_architectures/cv3ib5q
    # conv2d defaults to Relu with glorot init
    conv1_1 = ll.Conv2DLayer(inp_1, num_filters=32, filter_size=(3,3), pad='same')
    conv1_2 = ll.Conv2DLayer(inp_2, num_filters=32, filter_size=(3,3), pad='same', W=conv1_1.W, b=conv1_1.b)

    mpool1_1 = ll.Pool2DLayer(conv1_1, pool_size=3, mode='max')
    mpool1_2 = ll.Pool2DLayer(conv1_2, pool_size=3, mode='max')

    drop1_1 = ll.DropoutLayer(mpool1_1, p=0.5)
    drop1_2 = ll.DropoutLayer(mpool1_2, p=0.5)
    # now we're down to 128 / 3 = 42x42 inputs

    # 9/20: reducing the amount of filters in conv2 from 64->16 to hopefully deal with some overfitting
    # No bueno, reducing to 8 didn't help either
    conv2_1 = ll.Conv2DLayer(drop1_1, num_filters=4, filter_size=(3,3), pad='same')
    conv2_2 = ll.Conv2DLayer(drop1_2, num_filters=4, filter_size=(3,3), pad='same', W=conv2_1.W, b=conv2_1.b)

    mpool2_1 = ll.Pool2DLayer(conv2_1, pool_size=3, mode='max')
    mpool2_2 = ll.Pool2DLayer(conv2_2, pool_size=3, mode='max')

    drop2_1 = ll.DropoutLayer(mpool2_1, p=0.5)
    drop2_2 = ll.DropoutLayer(mpool2_2, p=0.5)
    # now we're down to 42 / 3 = 14x14 inputs, and 64 of them so we'll have 14x14x16 = 3136 inputs

    # for similarity, we'll do a few FC layers before merging

    fc1_1 = ll.DenseLayer(drop2_1, num_units=256)
    fc1_2 = ll.DenseLayer(drop2_2, num_units=256, W=fc1_1.W, b=fc1_1.b)


    fc2_1 = ll.DenseLayer(fc1_1, num_units=128, nonlinearity=linear)
    fc2_2 = ll.DenseLayer(fc1_2, num_units=128, nonlinearity=linear, W=fc2_1.W, b=fc2_1.b)

    distance_out = ll.ConcatLayer([fc2_1, fc2_2], axis=1) # 1 now the layer axis, 0 is batch axis, and 2 is feature axis

    # so for now let's assume that comparing the length 512 descriptor works


    return distance_out

def similarity_iter(output_layer, update_params):
    X1 = T.tensor4()
    X2 = T.tensor4()
    y = T.ivector()

    # find the input layers
    # TODO this better
    all_layers = ll.get_all_layers(output_layer)
    input_1 = filter(lambda x: x.name == 'input1', all_layers)[0]
    input_2 = filter(lambda x: x.name == 'input2', all_layers)[0]

    descriptors = ll.get_output(output_layer, {input_1: X1, input_2: X2})
    #descriptor_shape = ll.get_output_shape(output_layer, {input_1: X1, input_2: X2})
    #print("Network output shape: %r" % (descriptor_shape,))
    # distance minimization
    distance = (descriptors[:,0] - descriptors[:,1]).norm(2, axis=0)
    print(distance.shape)
    loss = T.mean(y*distance + (1 - y)*T.maximum(0, 1 - distance))
    all_params = ll.get_all_params(output_layer)
    updates = adam(loss, all_params, **update_params)

    train_iter = theano.function([X1, X2, y], loss, updates=updates, allow_input_downcast=True)
    theano.printing.pydotprint(loss, outfile='./loss_graph.png',var_with_name_simple=True)
    valid_iter = theano.function([X1, X2, y], loss)

    return {'train':train_iter, 'valid':valid_iter}

def train(iteration_funcs, dataset, batch_size=32):
    nbatch_train = dataset['train']['y'].shape[0] // batch_size
    nbatch_valid = dataset['valid']['y'].shape[0] // batch_size

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

def shuffle_dataset(dset):
    # assume dset has X1, X2, y
    X1, X2, y = shuffle(dset['X1'], dset['X2'], dset['y'])
    print("Post-shuffle shapes")
    print(X1.shape)
    print(X2.shape)
    print(y.shape)
    dset['X1'] = X1
    dset['X2'] = X2
    dset['y'] = y
    # TODO this more elegantly
    return dset


def main(dataset_name, n_epochs):
    original_dataset = load_dataset(join('/home/zj1992/windows/work2/datasets/Flukes/patches',dataset_name))
    all_datasets = dataset_prep(original_dataset)
    for patch_type in all_datasets:
        network = build_siamese_separate_similarity()
        iter_funcs = similarity_iter(network, {})
        for epoch in range(n_epochs):
            tic = time.time()
            print("%s: Epoch %d" % (patch_type, epoch))
            loss = train(iter_funcs, all_datasets[patch_type])
            # shuffle training set
            all_datasets[patch_type]['train'] = shuffle_dataset(all_datasets[patch_type]['train'])
            toc = time.time() - tic
            print(loss)
            print("Took %0.2f seconds" % toc)
        # TODO do something with the network once this is done

if __name__ == '__main__':
    try:
        dset_name = sys.argv[1]
        n_epochs = sys.argv[2]
    except IndexError:
        print("Usage: %s <dataset_name> <n_epochs>" % sys.argv[0])
        sys.exit(1)

    main(dset_name, int(n_epochs))
