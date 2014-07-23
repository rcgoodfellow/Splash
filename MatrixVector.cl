#include "SparseMatrix.h"

__kernel 
void 
matrix_vector_mul(unsigned int n,
      unsigned int N,
		  __global REAL *sm_values, 
		  __global unsigned int *row_sizes,
		  __global unsigned int *indices,
		  __global REAL *dv_values,
		  __global REAL *mv_values)
{
  int tid = get_global_id(0);
  int ri = tid * n, 
      rs = row_sizes[tid];

  mv_values[tid] = 0;
  for(int i=0; i<rs; ++i)
  {
      mv_values[tid] += sm_values[ri + i] * dv_values[indices[ri + i]];
  }
}
