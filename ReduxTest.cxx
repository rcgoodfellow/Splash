#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "API.h"
#include "Utility.hxx"
#include "Redux.hxx"

#include <random>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace splash;

#define SPLASHDIR "/home/ry/Splash/"

//data
REAL *x;

//random # generation stuff
random_device rd;
uniform_real_distribution<REAL> v_dst{0.5,10};
default_random_engine re{rd()};

//ocl stuff
cl::Platform platform;
vector<cl::Device> gpus;
cl::Context ctx;
cl::CommandQueue cqueue;
cl::Program::Sources sources;
string src_txt;
cl::Program splash_prog;
cl::Kernel kernel;
cl::Buffer b_x, b_r;

void gen_x(unsigned int sz) {
  x = (REAL*)malloc(sizeof(REAL)*sz);
  for(unsigned int i=0; i<sz; ++i) { x[i] = v_dst(re); }
}
 
void initOcl() {
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  platform = platforms[0];

  platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
  ctx = cl::Context(gpus);
  cqueue = cl::CommandQueue(ctx, gpus[0]);

  src_txt = read_file(SPLASHDIR "Redux.cl");
  sources = {
    make_pair(src_txt.c_str(), src_txt.length())
  };
  splash_prog = cl::Program(ctx, sources);
  try{ splash_prog.build("-I " SPLASHDIR " -DREAL=double"); }
  catch(cl::Error&) {
    throw runtime_error(splash_prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpus[0]));
  }
}

double c_sum(double *x, size_t sz) {
  
  auto start = high_resolution_clock::now();
  double res{0};
  for(size_t i = 0; i<sz; ++i) { res += x[i]; }
  auto end = high_resolution_clock::now();
  auto dt = duration_cast<milliseconds>(end-start);
  cout << "CPU performance" << endl;
  cout << dt.count() << " ms" << endl;
  
  return res;

}

void go()
{
  unsigned int N = 1e8;
  gen_x(N);
  initOcl();
  
  ReduxMem rxm = redux_alloc(x, N, ctx, gpus[0]);
  pair<cl::NDRange, cl::NDRange> exs = rdx_des_local_max()(N, gpus[0]);
  kernel = redux_add_kernel(splash_prog, rxm);

  cout << exs.first[0] << "," << exs.first[1] << endl;
  cout << exs.second[0] << "," << exs.second[1] << endl;

  double r;
  for(int i=0; i<1; ++i) {

    cout << "Enqueue kernel" << std::endl;
    cqueue.enqueueNDRangeKernel(kernel,
        cl::NullRange,
        exs.first,
        exs.second);

    cout << "Enqueue readback" << std::endl;
    cqueue.enqueueReadBuffer(rxm.grspace, CL_TRUE, 0, sizeof(REAL)*rxm.Ng, rxm.gs);

    r = 0;
    for(size_t i=0; i<rxm.Ng; ++i) { r += rxm.gs[i]; }
  
    double cr = c_sum(x, N);
    cout << "CPU Result: " << cr << endl;
    cout << "GPU Result: " << r << endl;
  }

}

int main() {

  go();
  return EXIT_SUCCESS;

}
