#include "API.h"
#include "Redux.hxx"
#include "Runtime.hxx"

#include <random>
#include <vector>
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace splash;

#define SPLASHDIR "/home/ry/Splash/"

//vector to be reduced
REAL *x;

//random number generation stuff
random_device rd;
uniform_real_distribution<REAL> v_dst{0.5,10};
default_random_engine re{rd()};

//OpenCL stuff
cl::Platform platform;
vector<cl::Device> gpus;
cl::Context ctx;
cl::CommandQueue cqueue;

//Splash OpenCL library loader class
LibSplash libsplash{SPLASHDIR};

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

int main() {

  unsigned int N = 1e6;
  gen_x(N);
  initOcl();

  ReduxC rc(x, N, ctx, gpus[0], 64, libsplash.get(ctx), ReduxC::Reducer::Add);

  cout << rc.G[0] << "," << rc.G[1] << endl;
  cout << rc.L[0] << "," << rc.L[1] << endl;

  for(int i=0; i<1; ++i) {

    double cr = c_sum(x, N);
    rc.execute(cqueue); 
    double r = rc.readback(cqueue);
    cout << "CPU Result: " << cr << endl;
    cout << "GPU Result: " << r << endl;

  }

  return EXIT_SUCCESS;

}
