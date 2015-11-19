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

//#define DLLEXPORT extern "C" __declspec(dllexport)

using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> NDArrayFlattened;
typedef Map<NDArrayFlattened> ExternNDArray;

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

void dtw(const ExternNDArray & seq1, const ExternNDArray & seq2,
         ExternNDArray & distmat_out, int window) {
    float dist;
    for (int i = 1; i < seq1.rows(); i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2.rows(), i + window); j++) {
        //for (int j = 1; j < seq2.rows(); j++) {
            dist = (seq1.row(i).array() - seq2.row(j).array()).matrix().norm();
            //cosi_dist = -1*cosine(seq1.row(i), seq2.row(j));
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
        }
    }
}

// No need for any Eigen stuff
extern "C" void dtw_scalar(void * seq1v, void * seq2v, int seq1_len, int seq2_len,
                int window, void * distmat_outv) {
    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    float* seq1 = (float*) seq1v;
    float* seq2 = (float*) seq2v;
    float* distmat_out = (float*) distmat_outv;
    //float* distance_mat = make_mat(seq1_len, seq2_len, INFINITY);
    clock_t tic = std::clock();
    //dtw(seq1, seq2, distmat_out);
    float dist;
    for (int i = 1; i < seq1_len; i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2_len, i + window); j++) {
        //for (int j = 1; j < seq2_len; j++) {
            dist = fabs(seq1[i] - seq2[j]);
            //cosi_dist = -1*cosine(seq1.row(i), seq2.row(j));
            distmat_out[int_pos(i,j,seq2_len)] = dist + 
                MIN(distmat_out[int_pos(i, j-1, seq2_len)], 
                    MIN(distmat_out[int_pos(i-1, j, seq2_len)],
                        distmat_out[int_pos(i-1, j-1, seq2_len)]));
        }
    }
    clock_t toc = std::clock();
    double elapsed = double(toc - tic) / CLOCKS_PER_SEC;
    printf("Took %0.5f seconds\n", elapsed);
}

/*
struct Point {
    int x, y;
};

struct FastDTWInfoObj {
    float distance;
    std::list<Point> path;
};

FastDTWInfoObj fastdtw_dtw(const MatrixXf & seq1, const MatrixXf & seq2,
                           const FastDTWInfoObj & info, bool usePath, 
                           int radius) {
    // if we need to construct the window from the path (i.e. usePath is true)
    // let's do that, otherwise 
    // iterate through the 'window' constructed  
}

// Largely a C++ reimplementation of https://github.com/slaypni/fastdtw/blob/master/fastdtw.py
void fastdtw(const MatrixXf & seq1, const MatrixXf & seq2,
             int radius) {
    int min_seq_size = radius + 2;
    if ((seq1.rows() < min_seq_size) || (seq2.rows() < min_seq_size)) {
        return dtw(seq1, seq2, distmat_out);
    }

    MatrixXf seq1_half = shrink(seq1);
    MatrixXf seq2_half = shrink(seq2);

    distance, path = fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
}
// EDIT on further review the Python implementation is poorly done, and redoing it from scratch in C++ 
// is not a good use of my time for the moment
*/
/*inline uint64_t rdtsc() {
        uint32_t low, high;
        asm volatile ("rdtsc" : "=a" (low), "=d" (high));
        return (uint64_t)high << 32 | low;
}*/

extern "C" void dtw_windowed(//void * seq1posv, void * seq2posv, 
                             void * seq1curvv, void * seq2curvv,
                             int seq1_len, int seq2_len, int window,
                             void * distmat_outv) {
    // THIS WILL SEGFAULT IF seq1 AND seq2 ARE NOT (n,m)x4
    //float* seq1 = (float*) seq1v;
    //float* seq2 = (float*) seq2v;
    //float* distance_mat = make_mat(seq1_len, seq2_len, INFINITY);
    //uint64_t tic = rdtsc();
    clock_t tic = std::clock();
    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    //MatrixXf distance_mat = MatrixXf::Constant(seq1_len, seq2_len, INFINITY);
    ExternNDArray distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    //distance_mat(0,0) = 0;
    //ExternNDArray seq1pos((float*)seq1posv ,seq1_len,2);
    ExternNDArray seq1curv((float*)seq1curvv ,seq1_len,2);

    //ExternNDArray seq2pos((float*)seq2posv ,seq2_len,2);
    ExternNDArray seq2curv((float*)seq2curvv ,seq2_len,2);

    dtw(seq1curv, seq2curv, distmat_out, window);
   
    clock_t toc = std::clock();
    double elapsed = double(toc - tic) / CLOCKS_PER_SEC;
    printf("Took %0.2f seconds\n", elapsed);
    //free(distance_mat);
    //return retval;
}

