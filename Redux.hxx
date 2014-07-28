#ifndef SPLASH_REDUX_HXX
#define SPLASH_REDUX_HXX
/******************************************************************************
 *  The Splash Project
 *
 *  reduction operations - 25 July '14
 *  ~ ry
 *
 *  This file contains host code for performing reduction operations on real
 *  numbers
 */

#include "API.h"
#include "Engine.hxx"
#include <cmath>
#include <stdexcept>

namespace splash {

/*= ReduxCData ================================================================
 *
 * The redux computation data object encapsulates a redux computation and 
 * associated data
 */


struct ReduxC {

  enum class Reducer { Add, Subtract, Multiply, Divide };
  Reducer reducer;

  //Input vector and associated OpenCL buffer
  REAL *x;
  cl::Buffer b_x;

  //Result vector and associated OpenCL buffer
  REAL *gs;
  cl::Buffer grspace;
  
  size_t 
    N,    //Input size.
    Nl,   //Local memory size.
    Ng,   //Global memory size.
    ipt;  //Items per thread.

  cl::NDRange
    G,    //Global range
    L;    //Local range

  //The kernel used for this computation
  cl::Kernel k;

  //Device in use for computation
  cl::Device dev;

  //Context in use for computation
  cl::Context ctx;

  //Splash opencl program (library)
  cl::Program libsplash;

  //Redux computation where @x originates on the host
  ReduxC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt, 
      cl::Program splashp, Reducer r);

  //Redux computation where @x is already on the GPU
  ReduxC(cl::Buffer x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp, Reducer r);
  
  void execute(cl::CommandQueue &q);
  void readback(cl::CommandQueue &q);
  ReduxC fullReduction(cl::CommandQueue &q);

  private:
    void computeThreadStrategy();
    void computeMemoryRequirements();
    void setOclMemory(bool alloc_bx);
    void initKernel();
    void setKernel();

};

}
#endif
