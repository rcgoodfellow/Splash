#include "SparseMatrix.hxx"

using namespace splash;
using std::vector;


void splash::smSet(SparseMatrix *M, std::vector<sm_element> elems) {
  for(const sm_element &e : elems) { sm_set(M, e.i, e.j, e.v); }
}

void splash::smSetRow(SparseMatrix *M, unsigned int i, 
    std::vector<sm_sub_element> elems) {
  for(const sm_sub_element &e : elems) { sm_set(M, i, e.i, e.v); }
}

void splash::smSetCol(SparseMatrix *M, unsigned int i, 
    std::vector<sm_sub_element> elems) {
  for(const sm_sub_element &e : elems) { sm_set(M, e.i, i, e.v); }
}

void splash::dvSet(DenseVector *v, std::vector<sm_sub_element> elems) {
  for(const sm_sub_element &e : elems) { dv_set(v, e.i, e.v); }
}
