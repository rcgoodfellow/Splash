#include "SparseMatrix.h"

#ifndef __OPENCL_VERSION__
SparseMatrix* create_EmptySparseMatrix(unsigned int N, unsigned int n)
{
  SparseMatrix *sm = (SparseMatrix*)malloc(sizeof(SparseMatrix));
  sm->N = N;
  sm->n = n;
  sm->row_sizes = (unsigned int*)malloc(sizeof(unsigned int)*N);
  memset(sm->row_sizes, 0, N * sizeof(unsigned int));
  sm->indices = (unsigned int*)malloc(sizeof(unsigned int)*N*n);
  memset(sm->indices, -1, N * n * sizeof(unsigned int));
  sm->values = (REAL*)malloc(sizeof(REAL)*N*n);
  memset(sm->values, 0, N * n * sizeof(REAL));
  return sm;
}

void destroy_SparseMatrix(SparseMatrix *M)
{
  free(M->row_sizes);
  M->row_sizes = NULL;
  free(M->indices);
  M->indices = NULL;
  free(M->values);
  M->values = NULL;
}

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

void sm_print(SparseMatrix *M)
{
  for(unsigned int i=0; i<M->N; ++i)
  {
    unsigned int rsz = M->row_sizes[i];
    printf("%u,%u\n", i, rsz);
    printf("[");
    for(unsigned int j=0; j<rsz; ++j)
    {
      printf("%d,", M->indices[i * M->n + j]);
    }
    printf("]\n");

    printf("[");
    for(unsigned int j=0; j<rsz; ++j)
    {
      printf("%f,", M->values[i * M->n + j]);
    }
    printf("]\n");
  }
}

#endif

unsigned int find(unsigned int *begin, unsigned int *end, unsigned int val)
{
  if(begin == end) { return 0; }
  unsigned int *start = begin;
  long pi = (end - begin) / 2;
  unsigned int *p = (unsigned int*)(begin + pi);

  while(end - begin > 1)
  {
    if(val <= *p) { end = p; }
    else { begin = p; }
    pi = (end - begin) / 2;
    p = (unsigned int*)(begin + pi);
  }

  if(val > *p){ ++p; }
  return p - start;
}

void mrshift(byte *begin, byte *end, unsigned int count, unsigned int size)
{
  for(byte *i = end; i >= begin; --i)
  {
    *(i + count * size) = *i;
  }
}

void mset(byte *data, unsigned int size, byte *tgt)
{
  for(unsigned int i=0; i<size; ++i) { tgt[i] = data[i]; }
}

void insert(byte *begin, byte *end, byte *val, unsigned int offset, 
    unsigned int size)
{
  mrshift(begin + offset * size, end, 1, size);
  mset(val, size, begin + offset * size);
}


void sm_set(SparseMatrix *M, unsigned int row, unsigned int col, REAL val)
{
  unsigned int *rb = &M->indices[row * M->n],
               *re = rb + M->row_sizes[row];
  REAL         *vb = &M->values[row * M->n],
               *ve = vb + M->row_sizes[row];

  int idx = find(rb, re, col);
  if(M->indices[M->n * row + idx] == col) { M->values[row * M->n + idx] = val; }
  else 
  { 
    insert((byte*)rb, (byte*)re, (byte*)&col, idx, sizeof(unsigned int));
    insert((byte*)vb, (byte*)ve, (byte*)&val, idx, sizeof(REAL));
    ++(M->row_sizes[row]);
  }
}

void dv_set(DenseVector *V, unsigned int row, REAL val)
{
  V->values[row] = val;
}


