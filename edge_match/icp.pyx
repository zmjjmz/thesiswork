#DEF NPY_NO_DEPRECATED_API = NPY_1_7_API_VERSION
# cython: profile=True
from __future__ import print_function, division
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
#from scipy.spatial.distance import cosine, euclidean


# Reference: http://docs.cython.org/src/userguide/numpy_tutorial.html#numpy-tutorial
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

# TODO: More generalizeable way to do this
cdef inline DTYPE_t float_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

cdef DTYPE_t decision_min(DTYPE_t above, DTYPE_t left, DTYPE_t match):
    return float_min(float_min(above, left), match)

#cdef inline DTYPE_t square(DTYPE_t a): return a*a

cdef DTYPE_t norm(DTYPE_t a, DTYPE_t b):
    return sqrt(pow(a,2) + pow(b,2))

cdef DTYPE_t dot(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

cdef DTYPE_t euclidean(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2):
    # assume (stupidly) that these are 2 vectors of x and y
    return norm((v1[0] - v2[0]), (v1[1] - v2[1]))
    #return np.linalg.norm(v1 - v2)

cdef DTYPE_t cosine(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2):
    return dot(v1, v2) / (norm(v1[0], v1[1]) * norm(v2[0], v2[1]))
    #return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

cdef DTYPE_t dtw_driver(np.ndarray[DTYPE_t, ndim=2] seq1, np.ndarray[DTYPE_t, ndim=2] seq2, int window):
    # each is an array of nx4, w/n elts and the first two columns are x, y coordinates while the second two columns are some measure of curvature
    # (differential for now)
    cdef np.ndarray[DTYPE_t, ndim=2] dtw = np.zeros((seq1.shape[0],seq2.shape[0]), dtype=DTYPE) + np.inf
    #dtw[:,0] = np.inf
    #dtw[0,:] = np.inf
    dtw[0,0] = 0
    window = int_max(window, abs(seq1.shape[0] - seq2.shape[0]))
    cdef int i = 0
    cdef int j = 0
    cdef DTYPE_t eucl_dist = 0
    cdef DTYPE_t curv_dist = 0
    for i in range(1,seq1.shape[0]):
        for j in range(int_max(1, i-window),int_min(seq2.shape[0],i+window)):
        #for j in range(seq2.shape[0]):
            #dist = np.linalg.norm(seq1[i,:] - seq2[j,:])
            eucl_dist = euclidean(seq1[i,:2], seq2[j,:2])
            curv_dist = cosine(seq1[i,2:] + 1e-7, seq2[j,2:] + 1e-7) # 0 curvature is a problem
            dtw[i,j] = (eucl_dist + curv_dist) + decision_min(dtw[i-1, j], dtw[i, j-1], dtw[i-1,j-1]) 
    return dtw[-1,-1]
    # backtrace from there

cpdef DTYPE_t dtw():
    seq1 = np.array(np.random.rand(800,4), dtype=np.float32)
    seq2 = np.array(np.random.rand(900,4), dtype=np.float32)
    window = 10
    return dtw_driver(seq1, seq2, window)

