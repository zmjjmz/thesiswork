import numpy as np
import glob
import lasagne
import lasagne.layers as ll
import theano
import theano.tensor as T
import time
import pickle
import random
from os.path import join, isdir, exists
from scipy.ndimage import imread
# Initialize weights from OverFeat
# Code lifted from https://github.com/sklearn-theano/sklearn-theano/blob/master/sklearn_theano/feature_extraction/overfeat.py
FILTER_SHAPES = np.array([(96, 3, 11, 11),
                            (256, 96, 5, 5),
                            (512, 256, 3, 3),
                            (1024, 512, 3, 3),
                            (1024, 1024, 3, 3),
                            (3072, 1024, 6, 6),
                            (4096, 3072, 1, 1),
                            (1000, 4096, 1, 1)])
BIAS_SHAPES = FILTER_SHAPES[:, 0]
def load_overfeat_weights(weightpath):
    # We'll just use the fast network for now
    print("Loading OverFeat weights")
    weightfile = join(weightpath, 'net_weight_0')
    memmap = np.memmap(weightfile, dtype=np.float32)
    mempointer = 0

    weights = []
    biases = []
    for weight_shape, bias_shape in zip(FILTER_SHAPES, BIAS_SHAPES):
        filter_size = np.prod(weight_shape)
        weights.append(
            memmap[mempointer:mempointer + filter_size].reshape(weight_shape))
        mempointer += filter_size
        biases.append(memmap[mempointer:mempointer + bias_shape])
        mempointer += bias_shape
    return weights, biases

def make_model(weights, biases, batchsize=1):
    print("Building model")
    inp_layer = ll.InputLayer(shape=(batchsize, 3, None, None)) # variable size for the FCN of course
    network_stack = [inp_layer]
    for layer_ind in range(len(weights)-1): # last layer is the softmax, will be treated differently
        # all conv layers
        print("Weight shapes")
        print(weights[layer_ind].shape)
        conv_layer = ll.Conv2DLayer(
                network_stack[layer_ind],
                num_filters = weights[layer_ind].shape[0],
                filter_size = tuple(weights[layer_ind].shape[2:4]),
                nonlinearity = lasagne.nonlinearities.rectify,
                W = weights[layer_ind],
                b = biases[layer_ind])
        network_stack.append(conv_layer)

    # now let's add the convolutional softmax layer. this final layer will be initialized randomly, and have to be tuned
    softmax = ll.Conv2DLayer(
            network_stack[-1],
            num_filters=2, # i.e. number of classes with the softmax nl
            filter_size=(1,1),
            nonlinearity=lasagne.nonlinearities.rectify, # oh boy
            W=lasagne.init.Uniform(),
            b=lasagne.init.Uniform())
    return softmax



def build_iterators(dset, network_output, batchsize=1, lr=0.1, momentum=0.3):
    # assume dset is a dict of 'train', 'test', 'valid' each being a tuple of X,y
    # build theano functions to evaluate each one
    index = T.iscalar('ind')
    X_b = T.btensor4('Xb') # each image is 3-D, and there's technically a batch size to take into consideration making it 4-D
    y_b = T.btensor4('yb') # output a 2-D image in an FCN, 1 channel for each class

    bslice = slice(index * batchsize, (index + 1) * batchsize)

    objf = lasagne.objectives.Objective(network_output, loss_function=lasagne.objectives.mse)

    train_loss = objf.get_loss(X_b, target=y_b)
    eval_loss = objf.get_loss(X_b, target=y_b, deterministic=True)

    pred_func = T.argmax(ll.get_output(network_output, X_b, deterministic=True), axis=1) # I think that should be the channel axis
    acc_func = T.mean(T.eq(pred_func, y_b), dtype=theano.config.floatX)

    # training schedule

    weights = lasagne.layers.get_all_params(network_output)
    updates = lasagne.updates.nesterov_momentum(
            train_loss, weights, lr, momentum)

    training_iterator = theano.function(
            [index], train_loss, updates=updates, givens={
                X_b: dset['train'][0][bslice],
                y_b: dset['train'][1][bslice]}
            )

    validation_iterator = theano.function(
            [index], [eval_loss, acc_func],
            givens={
                X_b: dset['valid'][0][bslice],
                y_b: dset['valid'][1][bslice]}
            )

    test_iterator = theano.function(
            [index], [eval_loss, acc_func],
            givens={
                X_b: dset['test'][0][bslice],
                y_b: dset['test'][1][bslice]}
            )
    return {
            'train':training_iterator,
            'valid':validation_iterator,
            'test':test_iterator,
    }


def train(iterators, dset, batchsize=1):
    nbatch_train = dset['train'][0].shape[0] // batchsize # 0th axis should always be the sample axis
    nbatch_valid = dset['valid'][0].shape[0] // batchsize

    for epoch in itertools.count(1):
        # generator for training
        batch_train_loss = []
        batch_valid_loss = []
        batch_valid_acc = []
        for batch_ind in range(nbatch_train):
            loss = iterators['train'](batch_ind) # where the magic happens
            batch_train_loss.append(loss)

        # only compute the validation after all of the training for the epoch is done
        for batch_ind in range(nbatch_valid):
            loss, acc = iterators['valid'](batch_ind)
            batch_valid_loss.append(loss)
            batch_valid_acc.append(acc)

        yield {
                'epoch_ind':epoch,
                'avg_train_loss': np.average(batch_train_loss),
                'avg_valid_loss': np.average(batch_valid_loss),
                'avg_valid_acc': np.average(batch_valid_acc),
        }


def _load_data(path):
    # for every indv / image, check if there is a segmentation folder
    # if there is, load the latest segmentationdef read_data(folder_root):
    """ Read through folder with format indv/images, collecting images and seg info """
    subfolder_list = glob.glob(join(path,'*'))
    indv_dict = {}
    seg_dict = {}
    for subfolder in subfolder_list:
        pictures = [i for i in glob.glob(join(subfolder,'*')) if not isdir(i)]
        indv = subfolder.split('/')[-1]
        indv_dict[indv] = pictures
        seg_dict[indv] = [_collect_segs(pic) for pic in pictures]

    return indv_dict, seg_dict

def _collect_segs(pic_fn):
    # lots of assumptions
    segfolder = pic_fn.split('.')[0] + '_segs'
    segmentations = []
    all_masks = glob.glob(join(segfolder,'*.mask.pkl'))
    #all_bgd = glob.glob(join(segfolder,'*.bgd.pkl'))
    #all_fgd = glob.glob(join(segfolder,'*.fgd.pkl'))
    # hacky but it should work so long as the filenames are well formed
    keys = ['mask','fgd','bgd']
    for i in range(len(all_masks)):
        segmentations.append({key:join(segfolder,"%d.%s.pkl" % (i,key)) for key in keys})
    return segmentations

def load_data_all(path):
    indv_dict, seg_dict = _load_data(path)
    final_X = []
    final_y = []
    for indv in indv_dict:
        for pic, segs in zip(indv_dict[indv], seg_dict[indv]):
            if len(segs) == 0:
                continue
            gt_seg_file = segs[-1]['mask']
            loaded_mask = pickle.load(open(gt_seg_file,'rb'))
            loaded_image = imread(pic).astype('float32')
            final_X.append(loaded_image)
            final_y.append(loaded_mask)
    train_ind = 7
    valid_ind = 9
    test_ind = 11
    split_final_X = {
            'train':final_X[0:train_ind],
            'valid':final_X[train_ind:valid_ind],
            'test':final_X[valid_ind:test_ind]
    }
    split_final_y = {
            'train':final_y[0:train_ind],
            'valid':final_y[train_ind:valid_ind],
            'test':final_y[valid_ind:test_ind],
    }
    print(split_final_X['train'].dtype)

    split_final_set = {}
    split_final_set['train'] = (theano.shared(split_final_X['train']), theano.shared(split_final_y['train']))
    split_final_set['valid'] = (theano.shared(split_final_X['valid']),theano.shared(split_final_y['valid']))
    split_final_set['test'] = (theano.shared(split_final_X['test']),theano.shared(split_final_y['test']))

    return split_final_set

if __name__ == '__main__':
    datapath = '/home/zj1992/windows/work2/datasets/humpbacks/'
    weightpath = '/home/zj1992/windows/work2/weights/overfeat/'
    dset = load_data_all(datapath)
    # quick & dirty train / valid / test split

    of_weights, of_biases = load_overfeat_weights(weightpath)
    overfeat_fcn = make_model(of_weights, of_biases)
    print(overfeat_fcn)
    iters = build_iterators(dset, overfeat_fcn)
    n_epochs = 10

    # now let's train each epoch
    now = time.time()
    try:
        for epoch in train(iters, dset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], n_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['avg_train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['avg_valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['avg_valid_acc'] * 100))

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass



























