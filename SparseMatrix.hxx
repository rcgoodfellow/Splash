#ifndef SPLASH_SPARSEMATRIX_HXX
#define SPLASH_SPARSEMATRIX_HXX

#include <vector>
#include "SparseMatrix.h"
#include "Vector.h"

namespace splash {

struct sm_element { 
  unsigned int i,j; REAL v; 
  sm_element() = default;
  sm_element(unsigned int i, unsigned int j, REAL v) : i{i}, j{j}, v{v} {}
};

struct sm_sub_element { 
  unsigned int i; REAL v; 
  sm_sub_element() = default;
  sm_sub_element(unsigned int i, REAL v) : i{i}, v{v} {}
};

void smSet(SparseMatrix *M, std::vector<sm_element> elems);
void smSetRow(SparseMatrix *M, unsigned int i, std::vector<sm_sub_element> elems);
void smSetCol(SparseMatrix *M, unsigned int i, std::vector<sm_sub_element> elems);

void dvSet(DenseVector *v, std::vector<sm_sub_element> elems);

}

#endif
