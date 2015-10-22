import pyximport
pyximport.install()
from icp import dtw
import numpy as np
import time

import cProfile, pstats

cProfile.runctx("dtw()", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

#random_seq1 = np.array(np.random.rand(800,4), dtype=np.float32)
#random_seq2 = np.array(np.random.rand(900,4), dtype=np.float32)


#tic = time.time()
#dist = dtw(random_seq1, random_seq2, 10)
#toc = time.time() - tic
#print("Result: %0.2f took %0.2f seconds" % (dist, toc))

