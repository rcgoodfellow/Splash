#include "API.h"

extern "C"{
#include "SparseMatrix.h"
#include "Utility.h"
}

#include "Runtime.hxx"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <vector>
#include <stdexcept>
#include <iostream>
#include <utility>

#include <CL/cl.h>
#include <CL/cl.hpp>

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
  size_t sm_sz, mv_sz;
  sm_source = read_file(SPLASHDIR "SparseMatrix.c", &sm_sz);
  mv_source = read_file(SPLASHDIR "MatrixVector.cl", &mv_sz);
  cl::Program::Sources src{
    std::make_pair(sm_source, sm_sz), 
    std::make_pair(mv_source, mv_sz)
  };

  const char *build_opt = "-I " SPLASHDIR;
  for(PlatformGroup &pg : pgroups) { 
    cl::Program *prog = pg.loadProgram(src, build_opt); 
    pg.loadKernel(prog, "matrix_vector_mul");
  } 

}

void loadData() {
  for(auto &pg : pgroups) {
    //the matrix itself
    pg.loadBuffer("M", 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        sizeof(REAL) * M->N * M->n,
        M->values);

    //row sizes
    pg.loadBuffer("M_ri",
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(unsigned int) * M->N,
        M->row_sizes);

    //indices
    pg.loadBuffer("M_idx",
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(unsigned int) * M->N * M->n,
        M->indices);

    //vector values
    pg.loadBuffer("v",
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(unsigned int) * v->N,
        v->values);

    //result vector
    pg.loadBuffer("Mv",
        CL_MEM_WRITE_ONLY,
        sizeof(REAL) * v->N,
        nullptr);
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
      q.enqueueNDRangeKernel(
          *k,
          cl::NullRange,
          cl::NDRange(M->N),
          cl::NDRange(1)
      );
    }
  }
}

void collectResults() {
  for(auto &pg : pgroups) {
    for(auto &q : pg.gqs) {
      q.enqueueReadBuffer(
          *pg.bufs.at("Mv"),
          CL_TRUE, //blocking read
          0, //offsett
          sizeof(REAL) * Mv->N,
          Mv->values);
    }
  }
}

void printResults()
{
  printf("Results *********************\n");
  for(unsigned int i=0; i<Mv->N; ++i)
  {
    printf("%f\n", Mv->values[i]);  
  }
}

int main()
{
  printf("%s", "Running basic linear algebra tests\n");

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

  initOclEnv();
  loadPrograms();
  loadData();
  setKernelArgs();
  enqueueKernels();
  collectResults();
  printResults();

  return EXIT_SUCCESS;
}

