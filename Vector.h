#ifndef SPLASH_VECTOR_H
#define SPLASH_VECTOR_H

#include "API.h"

typedef struct DenseVector
{
  unsigned int N;
  REAL *values;
}
DenseVector;

#ifndef _OPENCL_VERSION
API DenseVector* create_DenseVector(unsigned int N);
API void destroy_DenseVector(DenseVector *v);
#endif

API void dv_set(DenseVector *V, unsigned int row, REAL val);


API REAL
vector_norm_2(REAL *x, unsigned int N);

#endif
