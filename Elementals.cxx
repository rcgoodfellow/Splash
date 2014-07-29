#include "Elementals.hxx"

using namespace splash;
  
NormC::NormC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp) 

  : Computation{x, N, ipt, dev, ctx, splashp},
    redux{x, N, ctx, dev, ipt, libsplash, ReduxC::Reducer::Add} {

    computeShape();
    initMemory();
    initKernel();
    setKernel();
}
  
NormC::NormC(cl::Buffer b_x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt,
      cl::Program splashp)

  : Computation{b_x, N, ipt, dev, ctx, splashp},
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

  redux.execute(q);
  //--
  redux.readback(q);
  REAL r = *redux.result;
  std::cout << "expect: " << sqrt(r) << std::endl;
  //--

  b_x = redux.b_result;
  ksqrt.setArg(0, b_x);
  
  q.enqueueNDRangeKernel(ksqrt, cl::NullRange, ksqrt_shape.G, ksqrt_shape.L);
}
