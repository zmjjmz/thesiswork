import cPickle as pickle
from os.path import join

import lasagne.layers as ll
from lasagne.nonlinearities import linear
from lasagne.updates import adam
import theano.tensor as T

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

    # now we're down to 128 / 3 = 42x42 inputs

    conv2_1 = ll.Conv2DLayer(mpool1_1, num_filters=64, filter_size=(3,3), pad='same')
    conv2_2 = ll.Conv2DLayer(mpool1_2, num_filters=64, filter_size=(3,3), pad='same', W=conv2_1.W, b=conv2_1.b)

    mpool2_1 = ll.Conv2DLayer(conv2_1, pool_size=3, mode='max')
    mpool2_2 = ll.Conv2DLayer(conv2_2, pool_size=3, mode='max')

    # now we're down to 42 / 3 = 14x14 inputs, and 64 of them so we'll have 14x14x64 = 12544 inputs

    # for similarity, we'll do a few FC layers before merging

    fc1_1 = ll.DenseLayer(mpool2_1, num_units=1024)
    fc1_2 = ll.DenseLayer(mpool2_2, num_units=1024, W=fc1_1.W, b=fc1_1.b)


    fc2_1 = ll.DenseLayer(fc1_1, num_units=512, nonlinearity=linear)
    fc2_2 = ll.DenseLayer(fc2_2, num_units=512, nonlinearity=linear, W=fc2_1.W, b=fc2_1.b)

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

    # distance minimization
    distance = T.norm(2,descriptors[:,:,0] - descriptors[:,:,1],axis=1)
    loss = T.mean(y*distance + (1 - y)*T.maximum(0, 1 - distance))
    all_params = ll.get_all_params(output_layer)
    updates = adam(loss, all_params, **update_params)

    train_iter = theano.function([X1, X2, y], loss, updates=updates)
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

    avg_val_loss = np.mean(valid_losses)

    return {'train_loss':avg_train_loss,'valid_loss':avg_valid_loss}


def dataset_prep(original_dataset):
    # load the original dataset, which contains pairs of patchsets and whether or not they match
    # we want to get this into the form of three separate datasets, for each of left, right, notch
    # each of these will consist of two 4-tensors, the first axis being the sample axis, and the rest being (channel, width, height)
    # the y values will be whether or not X1[i] matches X2[i]

    # then organize this structure into a dictionary for each of train valid and test for each of the three, and train three models

def load_dataset(dataset_path):
    with open(

def main(dataset_name):
    original_dataset = join('/home/zj1992/windows/work2/datasets/Flukes/patches',dataset_name)

