#include "SparseMatrix.h"

__kernel
void
vector_norm_2(__global DenseVector *x, __global REAL *norm)
{
  int tid = get_global_id(0);
  
}
