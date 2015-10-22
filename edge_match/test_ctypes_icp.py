from __future__ import division
import numpy as np
import time
import ctypes
import random
from fastdtw import fastdtw
import cPickle as pickle
from scipy.spatial.distance import euclidean, cosine
from os.path import join

with open("../dataset_loc", 'r') as f:
    dataset_loc = f.read().rstrip()

print("Loading curvatures")
with open(join(dataset_loc, "Flukes/zooniverse_curvatures_winteg.pkl"), 'r') as f:
    curvatures = pickle.load(f)

lib = ctypes.cdll.LoadLibrary('./icp_ctypes_cpp/ctypes_icp.so')
dtw_dist = lib.dtw_windowed
#dtw_dist.restype = ctypes.c_float

niter = 10
cpptotoc = 0
fdtw_totoc = 0
for i in range(niter):
    # choose two random curvatures
    curvature1 = random.choice(curvatures)
    curvature2 = random.choice(curvatures)

    print("Comparing differential curvatures for id %s and id %s" % (curvature1['id'], curvature2['id']))
    #seq1_len = random.randint(800, 1000)
    #seq2_len = random.randint(700, 1000)
    seq1_len = curvature1['Q'].shape[0]
    seq2_len = curvature2['Q'].shape[0]
    distance_mat = (np.zeros((seq1_len, seq2_len), dtype=np.float32) + np.inf).flatten()
    distance_mat[0] = 0
    print(seq1_len)
    print(seq2_len)

    seq1curv = np.array(curvature1['int_curv'][5],dtype=np.float32).flatten()# + 1e-7
    seq2curv = np.array(curvature2['int_curv'][5],dtype=np.float32).flatten()# + 1e-7
    #seq2curv = np.random.rand(seq2_len,2).astype(np.float32).flatten()


    cpptic = time.time()
    distance = dtw_dist(
                        #ctypes.c_void_p(seq1pos.ctypes.data), ctypes.c_void_p(seq2pos.ctypes.data),
                        ctypes.c_void_p(seq1curv.ctypes.data), ctypes.c_void_p(seq2curv.ctypes.data),
                        ctypes.c_int(seq1_len), ctypes.c_int(seq2_len), ctypes.c_int(10),
                        ctypes.c_void_p(distance_mat.ctypes.data))
    print(distance_mat[-1])
    cpptotoc += time.time() - cpptic

    # now let's try this with fastdtw
    #fdtwtic = time.time()
    #fastdtw_dist = fastdtw(curvature1['int_curv'][5], curvature2['int_curv'][5], radius=20, dist=lambda a, b: euclidean(a,b))[0]
    #print(fastdtw_dist)
    #kfdtw_totoc += time.time() - fdtwtic

cpptotoc /= niter
#fdtw_totoc /= niter
print("C++ DTW Took %0.5f seconds on average over %d loops" % (cpptotoc, niter))
#print("Python FastDTW Took %0.5f seconds on average over %d loops" % (fdtw_totoc, niter))

