#include "Runtime.hxx"
#include "Vector.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

#define SPLASHDIR "/home/ry/Splash/"

using namespace std;
using namespace chrono;
using namespace splash;

PlatformGroup pg;

void initOCL() {
  pg = resolvePlatformGroups()[0];

  vector<string> ocl_srcs{
    SPLASHDIR "Vector.c",
    SPLASHDIR "Vector.cl"};

  string build_opt = string("-I ") + SPLASHDIR;
  cl::Program *p = pg.loadProgram(ocl_srcs, build_opt);
  pg.loadKernel(p, "k_vector_norm_2");

}

REAL* randomVector(size_t len)
{
  normal_distribution<REAL> dst{1,10};
  random_device rd;
  default_random_engine re{rd()};

  REAL *v = (REAL*)malloc(sizeof(REAL) * len);
  for(size_t i=0; i<len; ++i) { v[i] = dst(re); }

  return v;
}

REAL oclNorm(REAL *x, unsigned int N) {
  REAL nrm{0};
  cl::Buffer *bx = 
    pg.loadBuffer(to_string(N)+"x", CL_MEM_COPY_HOST_PTR, sizeof(REAL)*N, x);

  cl::Buffer *bnrm =
    pg.loadBuffer(to_string(N)+"n", CL_MEM_WRITE_ONLY, sizeof(REAL), nullptr);

  cl::Kernel *k = pg.kernels.at("k_vector_norm_2");
  k->setArg(0, *bx);
  k->setArg(1, N);
  k->setArg(2, *bnrm);

  pg.gqs[0].enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
  pg.gqs[0].enqueueReadBuffer(*bnrm, CL_TRUE, 0, sizeof(REAL), &nrm);
  return nrm;
}

template<class F>
void benchmark(F f, string s)
{
  cout << "measuring benchmark " << s << endl;
  auto start = high_resolution_clock::now();

  f();

  auto end  = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(end-start);
  cout << dt.count()/1000.0 << " us" << endl;
}

int main() {


  cout << "Building test vectors ...";
  vector<REAL*> inp{
    randomVector(1e3),
    randomVector(1e4),
    randomVector(1e5)
    //randomVector(1e6),
    //randomVector(1e7),
    //randomVector(1e8)
  };
  cout << "finished" << endl;
  
  initOCL();

  REAL n1, n1g; 
  benchmark([&inp, &n1](){ n1 = vector_norm_2(inp[0], 1e3); }, "host 1k");
  cout << "Host result: " << n1 << endl;

  benchmark([&inp, &n1g](){ n1g = oclNorm(inp[0], 1e3); }, "GPU 1k");
  cout << "GPU result: " << n1g << endl;

  REAL n10, n10g; 
  benchmark([&inp, &n10](){ n10 = vector_norm_2(inp[1], 1e4); }, "host 10k");
  cout << "Host result: " << n10 << endl;

  benchmark([&inp, &n10g](){ n10g = oclNorm(inp[1], 1e4); }, "GPU 10k");
  cout << "GPU result: " << n10g << endl;

}
