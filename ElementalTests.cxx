#include "Runtime.hxx"
#include "Elementals.hxx"
#include <iostream>

using namespace std;
using namespace splash;

#define SPLASHDIR "/home/ry/Splash/"

//Splash stuff
GPUEnv genv; //gpu environment
LibSplash libsplash{SPLASHDIR}; //splash opencl library loader

int main() {

  unsigned int N = 1e6;
  REAL *x = random_vector(N);

  NormC nc(x, N, genv.ctx, genv.dev, 64, libsplash.get(genv.ctx));
  nc.execute(genv.q);
  nc.readback(genv.q);
  REAL r = *nc.result;

  cout << "Result: " << r << endl;

}
