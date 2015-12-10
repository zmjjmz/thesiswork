from __future__ import division
import numpy as np
import time
import ctypes
import random
#from fastdtw import fastdtw
import cPickle as pickle
from scipy.spatial.distance import euclidean, cosine
from os.path import join
import utool as ut
import sys
from functools import partial
from itertools import product
import cv2

def block_integral_curvatures(sizes, coords):
    # assume coords are in x, y
    fit_size = (np.max(coords, axis=0) - np.min(coords, axis=0)) + (max(sizes)+1)
    binarized = np.zeros(fit_size[::-1])
    fixed_coords = (coords - np.min(coords, axis=0)) + max(sizes) // 2
    binarized[zip(*fixed_coords)[::-1]] = 1
    binarized = binarized.cumsum(axis=0)
    binarized[np.where(binarized > 0)] = 1
    summed_table = binarized.cumsum(axis=0).cumsum(axis=1)
    #plt.figure(figsize=(15,15))
    #plt.imshow(binarized)
    #print(np.sum(binarized))
    curvs = {}
    for size in sizes:
        curvs[size] = []
        for j, i in fixed_coords:
            starti = max(0, i- (size // 2))
            startj = max(0, j- (size // 2))
            endi = min(binarized.shape[0]-1, i+ (size//2))
            endj = min(binarized.shape[1]-1, j+ (size//2))
            this_summed_area = (summed_table[starti, startj] + summed_table[endi,endj] -
                                (summed_table[starti,endj] + summed_table[endi,startj]))
            this_summed_area /= size**2
            #this_summed_area = np.average(binarized[starti:endi+1,startj:endj+1])
            if this_summed_area < 1e-7:
                print("Warning: zero area!? ul: (%d, %d) lr: (%d, %d)" % (starti, startj, endi, endj))
            if np.isnan(this_summed_area):
                print("Empty slice: %s" % binarized[starti:endi+1,startj:endj+1])
                this_summed_area = 0.
            curvs[size].append(this_summed_area)
    return curvs

lib = ctypes.cdll.LoadLibrary('./icp_ctypes_cpp/ctypes_icp.so')
dtw_dist = lib.dtw_windowed
dtw_scalar = lib.dtw_scalar
dtw_hybrid = lib.dtw_hybrid
block_curv = lib.block_curvature

def block_integral_curvatures_cpp(sizes, coords):
    # assume coords are in x, y
    coords = np.array(coords, dtype=np.int32)
    fit_size = (np.max(coords, axis=0) - np.min(coords, axis=0)) + (max(sizes)+1)
    binarized = np.zeros(fit_size[::-1], dtype=np.float32)
    fixed_coords = np.ascontiguousarray((coords - np.min(coords, axis=0)) + max(sizes) // 2)[:,::-1]
    binarized[zip(*fixed_coords)] = 1
    binarized = binarized.cumsum(axis=0)
    binarized[np.where(binarized > 0)] = 1
    summed_table = binarized.cumsum(axis=0).cumsum(axis=1)
    curvs = {}

    coords_flat = fixed_coords.flatten()
    sat_flat = summed_table.flatten()
    for size in sizes:
        # compute curvature using separate calls to block_curv for each
        curvs[size] = np.zeros(fixed_coords.shape[0], dtype=np.float32)
        block_curv(ctypes.c_void_p(sat_flat.ctypes.data), ctypes.c_int(summed_table.shape[0]), ctypes.c_int(summed_table.shape[1]),
                   ctypes.c_void_p(coords_flat.ctypes.data), ctypes.c_int(fixed_coords.shape[0]), ctypes.c_int(size),
                   ctypes.c_void_p(curvs[size].ctypes.data))
    return curvs



def get_dist_mat_hybrid(query_pos, query_curv, db_pos, db_curv, curv_weights, alpha, window=50):
    ordered_sizes = sorted(curv_weights.keys())
    query_pos = np.array(query_pos, dtype=np.float32)
    db_pos = np.array(db_pos, dtype=np.float32)
    # we just need to stack the curvatures and make sure that the ordering is consistent w/curv_weights
    curv_weights_nd = np.array([curv_weights[i] for i in ordered_sizes],dtype=np.float32)
    query_curv_nd = np.hstack([np.array(query_curv[i],dtype=np.float32).reshape(-1,1) for i in ordered_sizes])
    db_curv_nd = np.hstack([np.array(db_curv[i],dtype=np.float32).reshape(-1,1) for i in ordered_sizes])

    query_len = query_pos.shape[0]
    db_len = db_pos.shape[0]
    distance_mat = (np.zeros((query_len, db_len), dtype=np.float32) + np.inf).flatten()
    distance_mat[0] = 0
    dtw_hybrid(
               ctypes.c_void_p(query_curv_nd.ctypes.data), ctypes.c_void_p(db_curv_nd.ctypes.data),
               ctypes.c_void_p(query_pos.ctypes.data), ctypes.c_void_p(db_pos.ctypes.data),
               ctypes.c_int(query_len), ctypes.c_int(db_len), ctypes.c_int(window),
               ctypes.c_int(curv_weights_nd.shape[0]), ctypes.c_void_p(curv_weights_nd.ctypes.data),
               ctypes.c_float(alpha), ctypes.c_void_p(distance_mat.ctypes.data))
    return distance_mat.reshape(query_len, db_len)


homog_aug = lambda x: np.hstack([x, np.ones((x.shape[0],1))])
def minimize_transform_affine(c1, c2):
    c1 = homog_aug(c1)
    #db_aug = np.hstack([db_aug, np.ones((db_aug.shape[0], 1))])

    transform = np.dot(np.linalg.inv(np.dot(c2.T, c2)),
                      np.dot(c2.T, c1))
    err = np.linalg.norm(c1 - np.dot(c2, transform))
    return transform, err

def hybrid_dist(c1, c2, compare_on=None, window=50, sizes=[5,10,15,20], alpha=0.5, weights=None, img_points_map=None):
    # align first
    if weights is None:
        weights = {size:1. for size in sizes}
    #transform, err = minimize_transform_affine(img_points_map[c1['fn']], img_points_map[c2['fn']])
    if img_points_map is None:
        print("No img_points_map provided, not aligning")
        aligned_c2 = c2['path']
    else:
        transform = cv2.getAffineTransform(img_points_map[c2['fn']], img_points_map[c1['fn']])
        aligned_c2 = np.ascontiguousarray(np.dot(transform, homog_aug(np.array(c2['path'])).T).T.astype(np.int32))
    # then get curvatures
    curv1 = block_integral_curvatures_cpp(sizes, c1['path'])
    curv2 = block_integral_curvatures_cpp(sizes, aligned_c2)
    # then get distance
    dist_mat = get_dist_mat_hybrid(c1['path'], curv1, aligned_c2, curv2,
                                   weights, alpha, window=window)
    return dist_mat[-1,-1]



def triplet_eval(curvatures, dist_method, compare_on, n_triplets=200, verbose=False):
    # generate triplets
    # the idea of a triplet is that we have an anchor sample and two other samples
    # one with the same id as the anchor and the other with a different id
    # (a1, a2, b) and we count a 0 if dist(a1,a2) < dist(a1,b)
    # this way we can evaluate a distance method using 2*n_triplets comparisons,
    # which is significantly faster
    id_curv_map = {}
    for curv in curvatures:
        if curv['id'] in id_curv_map:
            id_curv_map[curv['id']].append(curv)
        else:
            id_curv_map[curv['id']] = [curv]
    #print(len(id_curv_map))
    good_count = 0
    # for complete coverage we'll generate a triplet for at most each id
    idlist = filter(lambda x: len(id_curv_map[x]) > 1, id_curv_map.keys())
    for _ in ut.ProgressIter(range(n_triplets), label='triplet', enabled=verbose):
        anchor = random.choice(idlist)
        random.shuffle(id_curv_map[anchor])
        negative = random.choice(filter(lambda x: x != anchor, idlist))
        random.shuffle(id_curv_map[negative])
        anchor_pos_dist = dist_method(id_curv_map[anchor][0], id_curv_map[anchor][1],
                                     compare_on=compare_on)
        anchor_neg_dist = dist_method(id_curv_map[anchor][0], id_curv_map[negative][0],
                                     compare_on=compare_on)
        #print("Distance between two instances of %s: %0.2f" % (anchor, anchor_pos_dist))
        #print("Distance between %s and %s: %0.2f" % (anchor, negative, anchor_neg_dist))
        if anchor_pos_dist < anchor_neg_dist:
            good_count += 1

    return good_count / float(n_triplets)

if __name__ == "__main__":

    with open("../dataset_loc", 'r') as f:
        dataset_loc = f.read().rstrip()

    print("Loading curvatures")
    with open(join(dataset_loc, "Flukes/zooniverse_curvatures_winteg.pkl"), 'r') as f:
        curvatures = pickle.load(f)

    img_points_map = {}
    with open(join(dataset_loc, "Flukes/extracted_zsl_annotations.pkl"), 'r') as f:
        annot = pickle.load(f)
        for id_ in annot:
            for desc in annot[id_]:
                img_points_map[desc['fn']] = np.array([desc['left'], desc['right'], desc['notch']], dtype=np.float32)


    if len(sys.argv) < 3:
        print("Usage: %s n_triplets eval_save_file" % sys.argv[0])
        sys.exit(1)
    grid_windows = range(5,50,5)
    grid_alphas = np.linspace(0,1,10)
    best_window = None
    best_alpha = None
    best_eval = 0.5
    n_tripl = int(sys.argv[1])
    print("Grid searching over %d combinations of window and alpha" % (len(grid_windows)*len(grid_alphas)))
    window_alpha_grid = np.zeros((len(grid_windows), len(grid_alphas)))
    for (wind, window), (aind, alpha) in product(enumerate(grid_windows), enumerate(grid_alphas)):
        print("Evaluating window: %d, alpha: %0.1f on %d triplets" % (window, alpha, n_tripl))
        tic = time.time()
        this_eval = triplet_eval(curvatures, partial(hybrid_dist, alpha=alpha, window=window, img_points_map=img_points_map), 'path', n_triplets=n_tripl)
        window_alpha_grid[wind, aind] = this_eval
        toc = time.time() - tic
        print("Took %0.2f seconds, got an accuracy of %0.2f" % (toc, this_eval))
        if this_eval > best_eval:
            best_eval = this_eval
            best_alpha = alpha
            best_window = window

    print("Best accuracy was %0.2f with window of %d and alpha of %0.2f" % (best_eval, best_window, best_alpha))
    with open("%s.pkl" % sys.argv[2], 'w') as f:
        pickle.dump(window_alpha_grid, f)
    """
    tic = time.time()
    print(triplet_eval(curvatures, partial(hybrid_dist, alpha=0.33, window=45), 'path', n_triplets=int(sys.argv[1]), verbose=False))
    toc = time.time() - tic
    print("Took %0.2f seconds" % toc)
    niter = 1
    curv_tics = []
    hybrid_tics = []
    for i in range(niter):
        c1 = random.choice(curvatures)
        c2 = random.choice(curvatures)

        print("Comparing block curvs and coords for id %s (%d) and id %s (%d)" % (c1['id'], len(c1['path']), c2['id'], len(c2['path'])))
        sizes = [5,10,15,20]
        weights = {size:1. for size in sizes}
        alpha = 0.5
        window = 20

        # align paths
        #transform, err = minimize_transform_affine(img_points_map[c1['fn']], img_points_map[c2['fn']])
        #print("Alignment error: %0.2f" % err)
        transform = cv2.getAffineTransform(img_points_map[c2['fn']], img_points_map[c1['fn']])
        aligned_c2 = np.ascontiguousarray(np.dot(transform, homog_aug(np.array(c2['path'])).T).T.astype(np.int32))

        curv_tic = time.time()
        curv1 = block_integral_curvatures_cpp(sizes, c1['path'])
        curv2 = block_integral_curvatures_cpp(sizes, aligned_c2)
        curv_tics.append(time.time() - curv_tic)
        hybrid_tic = time.time()
        dist_mat = get_dist_mat_hybrid(c1['path'], curv1, aligned_c2, curv2,
                                       weights, alpha, window=window)
        hybrid_tics.append(time.time() - hybrid_tic)
        print(dist_mat[-1,-1])

    print("Getting curvatures: avg time %0.5f" % np.average(curv_tics))
    print("Getting distances: avg time %0.5f" % np.average(hybrid_tics))
    """
