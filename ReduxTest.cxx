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

void gen_x(unsigned int sz) {
  cout << "Generating input ...         " << flush;
  x = (REAL*)malloc(sizeof(REAL)*sz);
  for(unsigned int i=0; i<sz; ++i) { x[i] = v_dst(re); }
  cout << "OK" << endl;
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
  unsigned int N = 1e6;
  gen_x(N);
  initOcl();

  ReduxC rc(x, N, ctx, gpus[0], 64, splash_prog);

  cout << rc.G[0] << "," << rc.G[1] << endl;
  cout << rc.L[0] << "," << rc.L[1] << endl;

  for(int i=0; i<1; ++i) {

    double cr = c_sum(x, N);
    double r = rc.execute(cqueue); 
    cout << "CPU Result: " << cr << endl;
    cout << "GPU Result: " << r << endl;
  }

}

int main() {

  go();
  return EXIT_SUCCESS;

}
