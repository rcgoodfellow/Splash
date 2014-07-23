
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "API.h"

extern "C"{
#include "SparseMatrix.h"
}
#include "Utility.hxx"

#include "Runtime.hxx"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>





#define SPLASHDIR "/home/ry/Splash/"

using std::runtime_error;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::pair;

using namespace splash;

vector<PlatformGroup> pgroups;
char *sm_source{nullptr}, *mv_source{nullptr};

SparseMatrix *M;
DenseVector *v;
DenseVector *Mv; 

void initOclEnv() {
  pgroups = resolvePlatformGroups();

  size_t gpu_count{0};
  for(auto &pg : pgroups) { gpu_count += pg.gpus.size(); }

  cout << "Found " << pgroups.size() << " GPU platforms with "
    << gpu_count << " total GPUs" << endl;
}

void loadPrograms() {

  vector<string> srcs{
    SPLASHDIR "SparseMatrix.c",
    SPLASHDIR "MatrixVector.cl"};

  string build_opt = string("-I ") + SPLASHDIR;
  for(PlatformGroup &pg : pgroups) { 
    cl::Program *prog = pg.loadProgram(srcs, build_opt); 
    pg.loadKernel(prog, "matrix_vector_mul");
  } 

}

void loadData() {
  for(auto &pg : pgroups) {
    cl_mem_flags RC = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    //the matrix itself
    pg.loadBuffer("M", RC, sizeof(REAL) * M->N * M->n, M->values);
    //row sizes
    pg.loadBuffer("M_ri", RC, sizeof(unsigned int) * M->N, M->row_sizes);
    //indices
    pg.loadBuffer("M_idx", RC, sizeof(unsigned int) * M->N * M->n, M->indices);
    //vector values
    pg.loadBuffer("v", RC, sizeof(unsigned int) * v->N, v->values);
    //result vector
    pg.loadBuffer("Mv", CL_MEM_WRITE_ONLY, sizeof(REAL) * v->N, nullptr);
  }
}

void setKernelArgs() {
  for(auto &pg : pgroups) {
    cl::Kernel *k = pg.kernels.at("matrix_vector_mul");
    k->setArg(0, M->n);
    k->setArg(1, M->N);
    k->setArg(2, *pg.bufs.at("M"));
    k->setArg(3, *pg.bufs.at("M_ri"));
    k->setArg(4, *pg.bufs.at("M_idx"));
    k->setArg(5, *pg.bufs.at("v"));
    k->setArg(6, *pg.bufs.at("Mv"));
  }
}

void enqueueKernels() {
  for(auto &pg : pgroups) {
    cl::Kernel *k = pg.kernels.at("matrix_vector_mul");
    for(auto &q : pg.gqs) {
      q.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(M->N), 
          cl::NDRange(1));
    }}
}

void collectResults() {
  for(auto &pg : pgroups) { for(auto &q : pg.gqs) {
    q.enqueueReadBuffer(*pg.bufs.at("Mv"), CL_TRUE, 0, sizeof(REAL) * Mv->N,
        Mv->values);
  }}
}

void printResults() {
  printf("Results *********************\n");
  for(unsigned int i=0; i<Mv->N; ++i) { printf("%f\n", Mv->values[i]); }
}

void buildMatricies() {
  M = create_EmptySparseMatrix(5, 4);
  v = create_DenseVector(5);
  Mv = create_DenseVector(5);

  sm_set(M, 0, 0, 4);
  sm_set(M, 0, 1, 4);

  sm_set(M, 1, 0, 4);
  sm_set(M, 1, 3, 3);
  sm_set(M, 1, 2, 2);

  sm_set(M, 2, 3, 5);
  sm_set(M, 2, 2, 7);
  sm_set(M, 2, 1, 2);
 
  sm_set(M, 3, 2, 5);
  sm_set(M, 3, 1, 3);
  sm_set(M, 3, 4, 7);
  sm_set(M, 3, 3, 15);

  sm_set(M, 4, 3, 7);
  sm_set(M, 4, 4, 7);

  dv_set(v, 0, 1.2);
  dv_set(v, 1, 3.4);
  dv_set(v, 2, 5.6);
  dv_set(v, 3, 6.7);
  dv_set(v, 4, 7.8);
}

int main()
{
  printf("%s", "Running basic linear algebra tests\n");

  buildMatricies();
  initOclEnv();
  loadPrograms();
  loadData();
  setKernelArgs();
  enqueueKernels();
  collectResults();
  printResults();

  return EXIT_SUCCESS;
}

