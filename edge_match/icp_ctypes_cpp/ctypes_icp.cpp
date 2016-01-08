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
typedef Map<NDArrayFlattened> ExternNDArrayf;

typedef Matrix<int, Dynamic, Dynamic, RowMajor> NDArrayFlattenedi;
typedef Map<NDArrayFlattenedi> ExternNDArrayi;


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

void dtw(const ExternNDArrayf & seq1, const ExternNDArrayf & seq2,
         ExternNDArrayf & distmat_out, int window) {
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
    ExternNDArrayf distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    //distance_mat(0,0) = 0;
    //ExternNDArrayf seq1pos((float*)seq1posv ,seq1_len,2);
    ExternNDArrayf seq1curv((float*)seq1curvv ,seq1_len,2);

    //ExternNDArrayf seq2pos((float*)seq2posv ,seq2_len,2);
    ExternNDArrayf seq2curv((float*)seq2curvv ,seq2_len,2);

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
                const ExternNDArrayf & curv_hist_weights, float alpha,
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
    ExternNDArrayf distmat_out((float*)distmat_outv, seq1_len, seq2_len);

    ExternNDArrayf seq1pos((float*)seq1posv, seq1_len, 2);
    ExternNDArrayf seq2pos((float*)seq2posv, seq2_len, 2);

    ExternNDArrayf seq1curv((float*)seq1curvv, seq1_len, curv_hist_size);
    ExternNDArrayf seq2curv((float*)seq2curvv, seq2_len, curv_hist_size);


    ExternNDArrayf curv_hist_weights((float*)curv_hist_weightsv, curv_hist_size, 1);

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

extern "C" void dtw_curvweighted(void * seq1curvv, void* seq2curvv,
                                 int seq1_len, int seq2_len, int window,
                                 int curv_hist_size, void * curv_hist_weightsv,
                                 void * distmat_outv) {

    window = MAX(window, abs(seq1_len - seq2_len) + 1);
    ExternNDArrayf distmat_out((float*)distmat_outv, seq1_len, seq2_len);
    ExternNDArrayf seq1curv((float*)seq1curvv, seq1_len, curv_hist_size);
    ExternNDArrayf seq2curv((float*)seq2curvv, seq2_len, curv_hist_size);


    ExternNDArrayf curv_hist_weights((float*)curv_hist_weightsv, curv_hist_size, 1);

    float dist;
    for (int i = 1; i < seq1_len; i++) {
        for (int j = MAX(1, i - window); j < MIN(seq2_len, i + window); j++) {
            dist = ((seq1curv.row(i).array() * curv_hist_weights.transpose().array()) - 
                    (seq2curv.row(j).array() * curv_hist_weights.transpose().array())).matrix().norm();
            distmat_out(i,j) = dist + MIN(distmat_out(i, j-1), 
                                        MIN(distmat_out(i-1, j),
                                            distmat_out(i-1, j-1)));
        }
    }
}
extern "C" void block_curvature(void * summed_area_tabv, int binarized_rows, int binarized_cols,
                                void * seq_posv, int seq_len, 
                                int curvature_size, void * curvature_vecv) {

    ExternNDArrayf summed_area_tab((float*)summed_area_tabv, binarized_rows, binarized_cols);
    ExternNDArrayi seq_pos((int*)seq_posv, seq_len, 2);
    ExternNDArrayf curvature_vec((float*)curvature_vecv, seq_len, 1);
    float area = std::pow((float)curvature_size,2.0);

    for (int ind = 0; ind < seq_len; ind++) {
        // assume y, x
        int i = seq_pos(ind, 0);
        int j = seq_pos(ind, 1);
        int starti = MAX(0, i - (curvature_size / 2));
        int startj = MAX(0, j - (curvature_size / 2));
        int endi = MIN(binarized_rows - 1, i + (curvature_size / 2));
        int endj = MIN(binarized_cols - 1, j + (curvature_size / 2));
        //printf("i %d\n j %d\n", i, j);
        //printf("starti %d\nstartj %d\nendi %d\nendj %d\n", starti, startj, endi, endj);
        float this_summed_area = (float)(summed_area_tab(starti, startj) + summed_area_tab(endi, endj) -
                            (summed_area_tab(starti, endj) + summed_area_tab(endi, startj)));
        this_summed_area /= area; // may be a little wrong on the edges, but we shouldn't need to worry about that
        //printf("Point (%d, %d) has curvature %0.2f at size %d\n", i, j, this_summed_area, curvature_size);
        curvature_vec(ind) = this_summed_area; 
    }
}

float get_te_cost(int row, int col, int i, const MatrixXf & cost, const ExternNDArrayf & gradient_img) {
    if ((row + i < 0) || (row + i >= cost.rows()) || (col == 0)) {
        return INFINITY;
    } else {
        return cost(row+i, col-1) + gradient_img(row, col);
    }
}



extern "C" float find_trailing_edge(float * gradient_imgv, int gradient_rows, int gradient_cols,
                                   int startcol, int endrow, int endcol,
                                   int n_neighbors, int * outpathv) {
    ExternNDArrayf gradient_img(gradient_imgv, gradient_rows, gradient_cols);
    ExternNDArrayi outpath(outpathv, endcol - startcol, 2);
    /* Assume the gradient image is all setup, initialize cost and back */

    VectorXi neighbor_range(n_neighbors);
    printf("Building neighbor range\n");
    for (struct {int ind; int neighbor;} N = {0, (-1 * n_neighbors / 2)};
         N.neighbor<(n_neighbors / 2) + 1;
         N.neighbor++, N.ind++) {
        neighbor_range(N.ind,0) = N.neighbor;
    }
    MatrixXf cost = MatrixXf::Zero(gradient_rows, gradient_cols);
    MatrixXi back = MatrixXi::Zero(gradient_rows, gradient_cols);
    
    printf("Looping over image\n");
    for (int col = startcol; col <= endcol; col++) {
        for (int row = 0; row < gradient_rows; row++) {
            // argmin over candidates
            int best_candidate = 0; // middle
            float best_cand_cost = INFINITY;
            for (int i=0; i < neighbor_range.rows(); i++) {
                float cand_cost = get_te_cost(row, col, neighbor_range(i, 0), cost, gradient_img);
                if (cand_cost < best_cand_cost) {
                    best_candidate = neighbor_range(i, 0);
                    best_cand_cost = cand_cost;
                }
            }

            back(row, col) = best_candidate;
            cost(row, col) = best_cand_cost;
        }
    }
    // Now determine the optimal path from the endrow, endcol position
    // We'll store the result in outpath -- since we know how that the path is constructed 
    // One column at a time, we know how big the path will be ahead of time, which is very helpful
    int curr_row = endrow;
    float total_cost = 0;
    printf("Reconstructing the optimal path\n");
    for (struct {int ind; int col;} P = {0, endcol}; 
         P.col > startcol; P.col--, P.ind++) {
        total_cost += cost(curr_row, P.col);
        // x, y
        outpath(P.ind, 0) = P.col;
        outpath(P.ind, 1) = curr_row;

        curr_row = curr_row + back(curr_row, P.col);
    }

    return total_cost;

}
