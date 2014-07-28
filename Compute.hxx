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

//The local and global ranges from an N dimensional shape for a computation

struct Computation {

  struct Shape { 
    cl::NDRange G, L; 
    Shape(cl::NDRange G, cl::NDRange L) : G{G}, L{L} {} 
    Shape() = default;
  };
  
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

  Computation(cl::Device dev, cl::Context ctx, cl::Program l);

  virtual void execute(cl::CommandQueue &q) = 0;
  void readback(cl::CommandQueue &q);

};

struct VectorC : public Computation {

  //Input vector and associated OpenCL buffer
  REAL *x;
  cl::Buffer b_x;
  size_t N; //input size
  size_t ipt; //items per thread

  //true if the input is already on the gpu
  bool resident_input;

  VectorC(REAL *x, size_t N, size_t ipt, cl::Device dev, cl::Context ctx, 
      cl::Program l);
  
  VectorC(cl::Buffer b_x, size_t N, size_t ipt, cl::Device dev, cl::Context ctx,
      cl::Program l);
    
};

}

#endif
