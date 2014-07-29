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

#include "Engine.hxx"
#include <cmath>

namespace splash {

/*= ReduxC ====================================================================
 *
 * The redux computation object encapsulates a redux computation and 
 * associated data
 */
struct ReduxC : public VectorC {

  enum class Reducer { Add, Subtract, Multiply, Divide };
  Reducer reducer;

  cl::Kernel redux_kernel;
  Shape redux_shape;

  //Local memory size.
  size_t Nl;   

  //Redux computation where @x originates on the host
  ReduxC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt, 
      cl::Program splashp, Reducer r);

  //Redux computation where @x is already on the GPU
  ReduxC(cl::Buffer b_x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp, Reducer r);

  void execute(cl::CommandQueue &q);
  ReduxC fullReduction(cl::CommandQueue &q);

  private:
    //Compute the shape of the computation
    void computeReduxShape();

    //Initialize the memory required for the computation
    void initMemory();

    //initialize the redux kernel
    void initKernel();

    //Set the kernel args
    void setKernel();

};

}
#endif
