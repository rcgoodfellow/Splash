#ifndef _SPLASH_SPARSEMATRIX_
#define _SPLASH_SPARSEMATRIX_

#include "API.h"

typedef char byte;

#ifndef __OPENCL_VERSION__

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>

#endif

typedef struct SparseMatrix
{
  unsigned int N, n;
  unsigned int *row_sizes;
  unsigned int *indices;
  REAL *values;
}
SparseMatrix;

#ifndef __OPENCL_VERSION__

API SparseMatrix* create_EmptySparseMatrix(unsigned int N, unsigned int n);
API void destroy_SparseMatrix(SparseMatrix *M);

API void sm_print(SparseMatrix *M);

#endif

API void sm_set(SparseMatrix *M, unsigned int row, unsigned int col, REAL val);

API unsigned int find(unsigned int *begin, unsigned int *end, unsigned int val);
API void insert(byte *begin, byte *end, byte *val, unsigned int offset, 
    unsigned int size);
API void mrshift(byte *begin, byte *end, unsigned int count, unsigned int size);
API void mset(byte *data, unsigned int size, byte *tgt);

#endif
