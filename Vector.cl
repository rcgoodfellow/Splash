#include "Vector.h"

__kernel
void
k_vector_norm_2(__global REAL *x, unsigned int N, __global REAL *n) {
  if(get_global_id(0) == 0) {
    *n = vector_norm_2(x, N);
  }
}

