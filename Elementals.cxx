#include "Elementals.hxx"

using namespace splash;
  
NormC::NormC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp) 

  : VectorC{x, N, ipt, dev, ctx, splashp},
    redux{x, N, ctx, dev, ipt, libsplash, ReduxC::Reducer::Add} {

    computeShape();
    initMemory();
    initKernel();
    setKernel();
}
  
NormC::NormC(cl::Buffer b_x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp)

  : VectorC{b_x, N, ipt, dev, ctx, splashp},
    redux{x, N, ctx, dev, ipt, libsplash, ReduxC::Reducer::Add} {
    
    computeShape();
    initMemory();
    initKernel();
    setKernel();

}

void
NormC::computeShape() {

 ksqrt_shape = {{1,1}, {1,1}}; 

}

void
NormC::initMemory() {

  /*
  if(!resident_input) {
    b_x = cl::Buffer(
        ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(REAL),
        x);
  }
  */

  Nr = 1;
  b_result = cl::Buffer(
      ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(REAL),
      result);
}

void
NormC::initKernel() {
  ksqrt = cl::Kernel(libsplash, "ksqrt");
}

void
NormC::setKernel() {
  ksqrt.setArg(0, b_x);
  ksqrt.setArg(1, (unsigned long)1);
  ksqrt.setArg(2, (unsigned long)1);
  ksqrt.setArg(3, b_result);
}

#include <iostream>
void
NormC::execute(cl::CommandQueue &q) {

  ReduxC rc = redux.fullReduction(q);
  //--
  rc.readback(q);
  REAL r = *rc.result;
  std::cout << "expect: " << sqrt(r) << std::endl;
  //--

  b_x = rc.b_result;
  ksqrt.setArg(0, b_x);
  
  q.enqueueNDRangeKernel(ksqrt, cl::NullRange, ksqrt_shape.G, ksqrt_shape.L);
}
