#include "Redux.hxx"
#include "Runtime.hxx"

#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace splash;

#define SPLASHDIR "/home/ry/Splash/"

//Splash stuff
GPUEnv genv; //gpu environment
LibSplash libsplash{SPLASHDIR}; //splash opencl library loader

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
  cout << "Generating input ...         " << flush;
  REAL *x = random_vector(N);
  cout << "OK" << endl;

  ReduxC rc(x, N, genv.ctx, genv.dev, 64, libsplash.get(genv.ctx), 
      ReduxC::Reducer::Add);

  cout << rc.redux_shape.G[0] << "," << rc.redux_shape.G[1] << endl;
  cout << rc.redux_shape.L[0] << "," << rc.redux_shape.L[1] << endl;

  double cr = c_sum(x, N);
  ReduxC rc2 = rc.fullReduction(genv.q);
  rc2.readback(genv.q);
  double r = *rc2.result;

  cout << "CPU Result: " << cr << endl;
  cout << "GPU Result: " << r << endl;

  return EXIT_SUCCESS;
}
