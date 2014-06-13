#include "SparseMatrix.h"


__kernel void matrix_vector_mul(__global SparseMatrix *M, __global DenseVector *v, __global DenseVector *Mv)
{
    int tid = get_global_id(0);
    int ri = tid * M->n, 
        rs = M->row_sizes[tid];

    Mv->values[tid] = 0;
    for(int i=0; i<rs; ++i)
    {
        Mv->values[tid] += M->values[ri + i] * v->values[M->indices[ri + i]];
    }
}
