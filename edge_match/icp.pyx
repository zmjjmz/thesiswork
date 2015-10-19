from __future__ import print_function, division
#print("Dicks")

cdef dtw(seq1, seq2, int window):
    dtw = np.zeros((Q1.shape[0],Q2.shape[0])) + np.inf
    #dtw[:,0] = np.inf
    #dtw[0,:] = np.inf
    dtw[0,0] = 0
    window = max(window, abs(Q1.shape[0] - Q2.shape[0]))
    for i in range(1,Q1.shape[0]):
        for j in range(max(1, i-window),min(Q2.shape[0],i+window)):
            #dist = np.linalg.norm(Q1[i,:] - Q2[j,:])
            dist = cosine(Q1[i,:], Q2[j,:])
            dtw[i,j] = dist + min([dtw[i-1, j],
                                   dtw[i, j-1],
                                   dtw[i-1,j-1]])	
