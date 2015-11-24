#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
//#include <stdint.h>
#include <ctime>
//#include <cstdint>
#include <Eigen/Dense>
#include <iostream>
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

// No need for any Eigen stuff with just scalar comparisons
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

float norm_minmax(float max, float min, float x) {
    return (x - min) / (max - min);
}

float dist_func(const Array<float, 1, 2> & pos1, const Array<float, 1, 2> & pos2,
                const ArrayXf & curv1, const ArrayXf & curv2,
                const ExternNDArray & curv_hist_weights, float alpha,
                float eucl_max, float eucl_min
                ) {
    // alpha*(norm(pos1, pos2)) + (1-alpha)*(norm(elemwisemul(curv_hist_weights, curv1), elemwisemul(curv_hist_weights, curv2)))
    float dist = 0;
    dist += alpha * norm_minmax(eucl_max, eucl_min, (pos1 - pos2).matrix().norm());
    dist += (1 - alpha) * (((curv1 * curv_hist_weights.array()) - (curv2 * curv_hist_weights.array())).matrix().norm());
    return dist;
}

extern "C" void dtw_hybrid(void * seq1curvv, void* seq2curvv,
                           void * seq1posv, void * seq2posv,
                           int seq1_len, int seq2_len, int window,
                           int curv_hist_size, void * curv_hist_weightsv,
                           float alpha, void * distmat_outv) {

    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    ExternNDArray distmat_out((float*)distmat_outv, seq1_len, seq2_len);

    ExternNDArray seq1pos((float*)seq1posv, seq1_len, 2);
    ExternNDArray seq2pos((float*)seq2posv, seq2_len, 2);

    ExternNDArray seq1curv((float*)seq1curvv, seq1_len, curv_hist_size);
    ExternNDArray seq2curv((float*)seq2curvv, seq2_len, curv_hist_size);


    ExternNDArray curv_hist_weights((float*)curv_hist_weightsv, curv_hist_size, 1);

    int dist_counter = 0;
    std::clock_t dist_tocs = 0;
    // normalize
    // TODO: figure out better / faster way to do this
    float eucl_min = INFINITY;
    float eucl_max = -INFINITY;
    float eucl_dist; 
    for (int i = 1; i < seq1pos.rows(); i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2pos.rows(), i + window); j++) {
            eucl_dist = (seq1pos.row(i).array() - seq2pos.row(j).array()).matrix().norm();
            eucl_min = MIN(eucl_dist, eucl_min);
            eucl_max = MAX(eucl_dist, eucl_max);
        }
    }
    float dist;
    for (int i = 1; i < seq1pos.rows(); i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2pos.rows(), i + window); j++) {
            dist_counter += 1;
            std::clock_t disttic = std::clock();
            dist = dist_func(seq1pos.row(i).array(), seq2pos.row(j).array(), 
                             seq1curv.row(i).array(), seq2curv.row(j).array(), 
                             curv_hist_weights, alpha, eucl_max, eucl_min);
            /*
            if (i == 1 && j == 1) 
            {
                printf("Distance: %0.5f\n", dist);
                std::cout << "Seq 1 position: " << seq1pos.row(i) << " at ind " << i
                          << "\nSeq 2 position: " << seq2pos.row(j) << " at ind " << j 
                          << std::endl;
            }
            */
                
            //dist = (seq1.row(i).array() - seq2.row(j).array()).matrix().norm();
            //cosi_dist = -1*cosine(seq1.row(i), seq2.row(j));
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
            dist_tocs += std::clock() - disttic;
        }
    }

    //printf("%0.10f seconds (on avg) for %d distances\n", (dist_tocs / (double)dist_counter / (double)CLOCKS_PER_SEC), dist_counter);
}

