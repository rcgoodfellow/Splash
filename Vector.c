#include "Vector.h"

#ifndef __OPENCL_VERSION__
DenseVector* create_DenseVector(unsigned int N)
{
  DenseVector *v = (DenseVector*)malloc(sizeof(DenseVector));
  v->N = N;
  v->values = (REAL*)malloc(sizeof(REAL)*N);
  return v;
}

void destroy_DenseVector(DenseVector *v)
{
  free(v->values);
  v->values = NULL;
}
#endif

void dv_set(DenseVector *V, unsigned int row, REAL val)
{
  V->values[row] = val;
}


REAL
vector_norm_2(REAL *x, unsigned int N)
{
  REAL s = 0;
  for(unsigned int i=0; i<N; ++i) {
    s += x[i];
  }
  return sqrt(s);
}
