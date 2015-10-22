#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
//#include <stdint.h>
#include <ctime>
//#include <cstdint>
#include <Eigen/Dense>
//#include <cblas>
// this is bad don't do this kids
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define DLLEXPORT extern "C" __declspec(dllexport)

/*
extern "C" {
    float dtw_windowed(void * seq1v, void * seq2v, int seq1_len, int seq2_len, int window); 
}
*/

using namespace Eigen;
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
    return std::sqrt(square(a) + square(b));
}

inline float dot(const float a1, const float a2, const float b1, const float b2) {
    return a1*b1 + a2*b2;
}

inline float euclidean(const float a1, const float a2, const float b1, const float b2) {
    return norm(a1 - b1, a2 - b2);
}

inline float cosine(Matrix<float, 1, 2> a, Matrix<float, 1, 2> b) {
    return a.dot(b.transpose()) / (a.norm() * b.norm());
    //return dot(a1, a2, b1, b2) / (norm(a1, a2) * norm(b1, b2));
}

float* make_mat(const int n, const int m, const float fill) {
    float* mat = (float *)std::calloc(n * m, sizeof(float));
    int i, j; 
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            mat[int_pos(i, j, m)] = fill;
        }
    }
    return mat;
}

/*inline uint64_t rdtsc() {
        uint32_t low, high;
        asm volatile ("rdtsc" : "=a" (low), "=d" (high));
        return (uint64_t)high << 32 | low;
}*/
typedef Matrix<float, Dynamic, Dynamic, RowMajor> NDArrayFlattened;

DLLEXPORT void dtw_windowed(//void * seq1posv, void * seq2posv, 
                             void * seq1curvv, void * seq2curvv,
                             int seq1_len, int seq2_len, int window,
                             void * distmat_outv) {
    // THIS WILL SEGFAULT IF seq1 AND seq2 ARE NOT (n,m)x4
    //float* seq1 = (float*) seq1v;
    //float* seq2 = (float*) seq2v;
    //float* distance_mat = make_mat(seq1_len, seq2_len, INFINITY);
    //uint64_t tic = rdtsc();
    clock_t tic = std::clock();
    //MatrixXf distance_mat = MatrixXf::Constant(seq1_len, seq2_len, INFINITY);
    Map<NDArrayFlattened> distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    //distance_mat(0,0) = 0;
    //Map<NDArrayFlattened> seq1pos((float*)seq1posv ,seq1_len,2);
    Map<NDArrayFlattened> seq1curv((float*)seq1curvv ,seq1_len,2);

    //Map<NDArrayFlattened> seq2pos((float*)seq2posv ,seq2_len,2);
    Map<NDArrayFlattened> seq2curv((float*)seq2curvv ,seq2_len,2);

    //printf("n: %d, m: %d\n", seq1_len, seq2_len);
    window = MAX(window, abs(seq1_len - seq2_len));
    //printf("Window: %d\n", window);
    int i, j;
    float dist;
    for (i = 1; i < seq1_len; i++) {
        //for (j = MAX(1, i-window); j < MIN(seq2_len,i+window); j++) {
        for (j = 1; j < seq2_len; j++) {
            // Wish there was a better way to do this...
            dist = (seq1curv.row(i).array() - seq2curv.row(j).array()).matrix().norm();
            //cosi_dist = -1*cosine(seq1curv.row(i), seq2curv.row(j));
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
        }   
    }
    // find minimal value along the last column as our distance
    //float retval = distance_mat(seq1_len-1, seq2_len-1);
    //uint64_t toc = rdtsc() - tic;
    clock_t toc = std::clock();
    double elapsed = double(toc - tic) / CLOCKS_PER_SEC;
    printf("Took %0.2f seconds\n", elapsed);
    //free(distance_mat);
    //return retval;
}

//DLLEXPORT void fastdtw  TODO implement fastdtw 

