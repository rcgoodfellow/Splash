//#include "Runtime.hxx"
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "Vector.h"
#include "SparseMatrix.hxx"
#include "Utility.hxx"

#include <iostream>
#include <chrono>
#include <vector>
#include <random>

#define SPLASHDIR "/home/ry/Splash/"

using namespace std;
using namespace chrono;
using namespace splash;

//PlatformGroup pg;
SparseMatrix *M;
REAL *x, *cMx, *gMx;
cl::Platform platform;
vector<cl::Device> cpus, gpus;
cl::Context cpu_ctx, gpu_ctx;
cl::CommandQueue cpuq, gpuq;
cl::Program::Sources src;
string sm_cl_src;
cl::Program cpu_prog, gpu_prog;
cl::Kernel cpu_kernel, gpu_kernel;

cl::Buffer c_smv, c_rs, c_i, c_dv, c_mv,
           g_smv, g_rs, g_i, g_dv, g_mv;
  
random_device rd;
uniform_real_distribution<REAL> v_dst{0.5,10};
default_random_engine re{rd()};

void buildRandomMatrix(unsigned int N, unsigned int n) {
  M = create_EmptySparseMatrix(N, n);
  uniform_int_distribution<unsigned int> N_dst{0,N};
  uniform_int_distribution<unsigned int> n_dst{1,n};
  for(size_t i=0; i<N; ++i) {
    for(size_t j=0; j<n_dst(re); ++j) {
      sm_set(M, i, N_dst(re), v_dst(re)); 
  }}
}

void buildRandomVector(unsigned int N) {
  x = (REAL*)malloc(sizeof(REAL)*N);
  for(size_t i=0; i<N; ++i) { x[i] = v_dst(re); }
}

void buildRandomSystem(unsigned int N, unsigned int n) {
   buildRandomMatrix(N, n);
   buildRandomVector(N);
}

void initOcl() {
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  platform = platforms[0];

  platform.getDevices(CL_DEVICE_TYPE_CPU, &cpus);
  platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
  
  cpu_ctx = cl::Context(cpus);
  gpu_ctx = cl::Context(gpus);

  cpuq = cl::CommandQueue(cpu_ctx, cpus[0]);
  gpuq = cl::CommandQueue(gpu_ctx, gpus[0]);

  sm_cl_src = read_file(SPLASHDIR "MatrixVector.cl");
  src = {
    make_pair(sm_cl_src.c_str(), sm_cl_src.length())
  };

  cpu_prog = cl::Program(cpu_ctx, src);
  gpu_prog = cl::Program(gpu_ctx, src);
  try{ 
    cpu_prog.build("-I " SPLASHDIR); 
    gpu_prog.build("-I " SPLASHDIR);
  }
  catch(cl::Error&) {
    throw runtime_error(cpu_prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cpus[0]));
  }

  cpu_kernel = cl::Kernel(cpu_prog, "matrix_vector_mul");
  gpu_kernel = cl::Kernel(gpu_prog, "matrix_vector_mul");

  cl_mem_flags RC = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
  c_smv = cl::Buffer(cpu_ctx, RC, sizeof(REAL) * M->N * M->n, M->values);
  c_rs = cl::Buffer(cpu_ctx, RC, sizeof(unsigned int) * M->N, M->row_sizes);
  c_i = cl::Buffer(cpu_ctx, RC, sizeof(unsigned int) * M->N * M->n, M->indices);
  c_dv = cl::Buffer(cpu_ctx, RC, sizeof(REAL) * M->N, x);
  c_mv = cl::Buffer(cpu_ctx, CL_MEM_WRITE_ONLY, sizeof(REAL) * M->N, nullptr);

  g_smv = cl::Buffer(gpu_ctx, RC, sizeof(REAL) * M->N * M->n, M->values);
  g_rs = cl::Buffer(gpu_ctx, RC, sizeof(unsigned int) * M->N, M->row_sizes);
  g_i = cl::Buffer(gpu_ctx, RC, sizeof(unsigned int) * M->N * M->n, M->indices);
  g_dv = cl::Buffer(gpu_ctx, RC, sizeof(REAL) * M->N, x);
  g_mv = cl::Buffer(gpu_ctx, CL_MEM_WRITE_ONLY, sizeof(REAL) * M->N, nullptr);

  cpu_kernel.setArg(0, M->n);
  cpu_kernel.setArg(1, M->N);
  cpu_kernel.setArg(2, c_smv);
  cpu_kernel.setArg(3, c_rs);
  cpu_kernel.setArg(4, c_i);
  cpu_kernel.setArg(5, c_dv);
  cpu_kernel.setArg(6, c_mv);

  gpu_kernel.setArg(0, M->n);
  gpu_kernel.setArg(1, M->N);
  gpu_kernel.setArg(2, g_smv);
  gpu_kernel.setArg(3, g_rs);
  gpu_kernel.setArg(4, g_i);
  gpu_kernel.setArg(5, g_dv);
  gpu_kernel.setArg(6, g_mv);

  cMx = (REAL*)malloc(sizeof(REAL) * M->N);
  gMx = (REAL*)malloc(sizeof(REAL) * M->N);

  cout << "measuring cpu performance" << endl;

  cpuq.enqueueNDRangeKernel(cpu_kernel, 
      cl::NullRange, cl::NDRange(M->N), cl::NDRange(32));
  cpuq.enqueueReadBuffer(c_mv, CL_TRUE, 0, sizeof(REAL) * M->N, cMx);
  auto start = high_resolution_clock::now();
  for(int i=0; i<15; ++i) {
    cpuq.enqueueNDRangeKernel(cpu_kernel, 
        cl::NullRange, cl::NDRange(M->N), cl::NDRange(32));
    cpuq.enqueueReadBuffer(c_mv, CL_TRUE, 0, sizeof(REAL) * M->N, cMx);
  }

  auto end  = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(end-start);
  cout << dt.count()/1000.0 << " us" << endl;
  
  cout << "measuring gpu performance" << endl;
  
  gpuq.enqueueNDRangeKernel(gpu_kernel, 
      cl::NullRange, cl::NDRange(M->N), cl::NDRange(256));
  gpuq.enqueueReadBuffer(g_mv, CL_TRUE, 0, sizeof(REAL) * M->N, gMx);
  start = high_resolution_clock::now();
  for(int i=0; i<15; ++i) {
    gpuq.enqueueNDRangeKernel(gpu_kernel, 
        cl::NullRange, cl::NDRange(M->N), cl::NDRange(256));
    gpuq.enqueueReadBuffer(g_mv, CL_TRUE, 0, sizeof(REAL) * M->N, gMx);
  }
  
  end  = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(end-start);
  cout << dt.count()/1000.0 << " us" << endl;

  cout << "results cpu:~:gpu" << endl;
  for(unsigned int i=0; i<10; ++i) {
    cout << cMx[i] << ":~:" << gMx[i] << endl;
  }
}

int main() {
 
  unsigned int N = 100000;
  unsigned int n = ceil(N*0.00035);
  cout << "Data Size: " << (N * n * sizeof(REAL))/1024.0/1024.0 << "MB" << endl;
  cout << "Building system" << endl;
  buildRandomSystem(N, n);
  cout << "Done" << endl;
  //sm_print(M);

  try{ initOcl(); }
  catch(cl::Error &e) {
    throw runtime_error(e.what() + string(" ") + to_string(e.err()));
  }

  return EXIT_SUCCESS;
}

