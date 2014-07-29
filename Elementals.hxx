#ifndef SPLASH_ELEMENTALS_HXX
#define SPLASH_ELEMENTALS_HXX

/******************************************************************************
 *  The Splash Project
 *
 *  Elemenal computations for vectors - 28 July '14
 *  ~ ry
 *
 *  This file contains host code for performing elemental computations on real
 *  vectors
 */

#include "Engine.hxx"
#include "Redux.hxx"
#include <stdexcept>
#include <cmath>

namespace splash {

/*= NormC =====================================================================
 *
 * The norm computation object encapsulates a redux computation and associated
 * data
 */


struct NormC : public VectorC {

  ReduxC redux;

  cl::Kernel ksqrt;
  Computation::Shape ksqrt_shape;

  NormC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp);
  
  NormC(cl::Buffer b_x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp);

  void execute(cl::CommandQueue &q);

  private:
    void computeShape();
    void initMemory();
    void initKernel();
    void setKernel();
};

}

#endif
