#ifndef SPLASH_COMPUTE_HXX
#define SPLASH_COMPUTE_HXX
/******************************************************************************
 *  The Splash Project
 *
 *  Splash runtime engine computation components - 28 July '14
 *
 *  This file contains definitions used by the splash engine to keep track
 *  of OpenCL computations.
 */
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include "API.h"

namespace splash {

struct Computation {

  //Nested Types ==============================================================

  enum class IOType { Vector, Scalar };

  //The local and global ranges from an N dimensional shape for a computation
  struct Shape { 
    cl::NDRange G, L; 
    Shape(cl::NDRange G, cl::NDRange L) : G{G}, L{L} {} 
    Shape() = default;
  };

  // ==========================================================================

  IOType input_iot, output_iot;
  
  //Input vector and associated OpenCL buffer
  REAL *x;
  cl::Buffer b_x;
  
  //size of input
  size_t Ni;
  
  //items per thread
  size_t ipt; 

  //size of the result
  size_t Nr; 

  //Device in use for computation
  cl::Device dev;

  //Context in use for computation
  cl::Context ctx;

  //Splash opencl program (library)
  cl::Program libsplash;
  
  //Result vector and associated OpenCL buffer
  REAL *result;
  cl::Buffer b_result;
  
  //true if the input is already on the gpu
  bool resident_input;

  Computation(REAL *x, size_t Ni, size_t ipt, 
      cl::Device dev, cl::Context ctx, cl::Program l);

  Computation(cl::Buffer b_x, size_t Ni, size_t ipt, 
      cl::Device dev, cl::Context ctx, cl::Program l);

  virtual void execute(cl::CommandQueue &q) = 0;
  void readback(cl::CommandQueue &q);

};

}

#endif
