#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "float.h"
#include "cblas.h"
// this is bad don't do this kids
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int int_pos(int i, int j, int cols) {
    // row major
    //printf("Access: %d*%d + %d = %d\n", i, cols, j, i*cols +j);
    return i*cols + j;
}

inline float square(const float a) {
    return a*a;
}

inline float norm(const float a, const float b) {
    //return sqrt(pow(a, 2) + pow(b, 2));
    return sqrt(square(a) + square(b));
}

inline float dot(const float a1, const float a2, const float b1, const float b2) {
    return a1*b1 + a2*b2;
}

inline float euclidean(const float a1, const float a2, const float b1, const float b2) {
    return norm(a1 - b1, a2 - b2);
}

inline float cosine(const float a1, const float a2, const float b1, const float b2) {
    return dot(a1, a2, b1, b2) / (norm(a1, a2) * norm(b1, b2));
}

float* make_mat(const int n, const int m, const float fill) {
    float* mat = (float *)calloc(n * m, sizeof(float));
    int i, j; 
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            mat[int_pos(i, j, m)] = fill;
        }
    }
    return mat;
}

float dtw_windowed(void * seq1v, void * seq2v, int seq1_len, int seq2_len, int window) {
    // THIS WILL SEGFAULT IF seq1 AND seq2 ARE NOT (n,m)x4
    float* seq1 = (float*) seq1v;
    float* seq2 = (float*) seq2v;
    float* distance_mat = make_mat(seq1_len, seq2_len, INFINITY);
    //printf("n: %d, m: %d\n", seq1_len, seq2_len);
    distance_mat[int_pos(0,0, seq1_len)] = 0;
    window = MAX(window, abs(seq1_len - seq2_len));
    //printf("Window: %d\n", window);
    int i, j;
    for (i = 1; i < seq1_len; i++) {
        //for (j = MAX(1, i-window); j < MIN(seq2_len,i+window); j++) {
        for (j = 1; j < seq2_len; j++) {
            // Wish there was a better way to do this...
            float eucl_dist = euclidean(seq1[int_pos(i,0,4)], seq1[int_pos(i,1,4)], seq2[int_pos(j,0,4)], seq2[int_pos(j,1,4)]);
            float cosi_dist = cosine(seq1[int_pos(i,2,4)], seq1[int_pos(i,3,4)], seq2[int_pos(j,2,4)], seq2[int_pos(j,3,4)]);
            distance_mat[int_pos(i,j,seq2_len)] = (eucl_dist + cosi_dist) + MIN(distance_mat[int_pos(i-1,j-1,seq2_len)], 
                                                                                MIN(distance_mat[int_pos(i-1,j,seq2_len)],
                                                                                    distance_mat[int_pos(i,j-1,seq2_len)]));
        }   
    }
    // find minimal value along the last column as our distance
    float retval = distance_mat[int_pos(seq1_len-1, seq2_len-1, seq2_len)];
    free(distance_mat);
    return retval;
}
