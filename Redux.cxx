#include "Redux.hxx"

using namespace splash;
using std::pair;
using std::runtime_error;

#include <iostream>

void
ReduxC::initMemory() {
 
  //Create an OpenCL buffer for the input
  if(!resident_input) {
    b_x = cl::Buffer(
        ctx, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
        sizeof(REAL) * N, 
        x);
  }

  //The result memory is the number of required workgroups becuase each
  //workgroup eventually writes its result back to a point in the global 
  //array. Since the redux kernel is a 2 dimensional kernel, the total 
  //number of workgroups is computed as the product of the number of 
  //workgroups in each dimension.
  Nr = (redux_shape.G[0] / redux_shape.L[0]) * (redux_shape.G[1] / redux_shape.L[1]);

  /**** no allocation is necessary here because local kernel memory is 
        allocated by the opencl runtime driver ****/

  //The local memory is computed as the size of the workgroup. Each
  //thread within a workgroup reduces @elem_per_pe elements onto a single
  //point in the LDS, thus the LDS must be the size of the workgroup
  Nl = redux_shape.L[0] * redux_shape.L[1];
  
  result = (REAL*)malloc(sizeof(REAL)*Nr);
  b_result = cl::Buffer(
      ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(REAL) * Nr,
      result);

}

void
ReduxC::computeReduxShape() {
  
    //The local execution range is the maximum square (2 dimensions)
    size_t 
      max_local = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
      L0 = sqrt(max_local),
      L1 = L0;

    //The total number of threads is the size of the input divided by the
    //number of elements each thread will consume.
    size_t total_thds = ceil(N / (float)ipt);
    
    //Turn the total number of threads into a square allocation
    size_t
      G0 = sqrt(total_thds),
      G1 = G0;

    //Ensure that each dimension of the square is a multiple of the
    //corresponding local execution range (OpenCL requirement for
    //defined behavior).
    G0 += (L0 - G0 % L0);
    G1 += (L1 - G1 % L1);
   
    redux_shape = {{G0, G1}, {L0, L1}};

}

void
ReduxC::initKernel() {

  switch(reducer) {
    case Reducer::Add : redux_kernel = cl::Kernel(libsplash, "redux_add"); break;
    default: throw runtime_error("Not Implemented");
  }

}

void
ReduxC::setKernel() {

  //See Redux.cl to see how this lines up
  redux_kernel.setArg(0, b_x);
  redux_kernel.setArg(1, N);
  redux_kernel.setArg(2, ipt);
  redux_kernel.setArg(3, cl::Local(Nl * sizeof(REAL)));
  redux_kernel.setArg(4, b_result);

}

ReduxC::ReduxC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, size_t ipt, 
    cl::Program libsplash, Reducer r)
  
  : VectorC{x, N, ipt, dev, ctx, libsplash},
    reducer{r} {
 
    computeReduxShape();
    initMemory();
    initKernel();
    setKernel();

}

ReduxC::ReduxC(cl::Buffer b_x, size_t N, cl::Context ctx, cl::Device dev, 
    size_t ipt, cl::Program libsplash, Reducer r)
  
  : VectorC{b_x, N, ipt, dev, ctx, libsplash}, 
    reducer{r} {

    computeReduxShape();
    initMemory();
    initKernel();
    setKernel();
}

void 
ReduxC::execute(cl::CommandQueue &q) {
  
  q.enqueueNDRangeKernel(redux_kernel, cl::NullRange, redux_shape.G, redux_shape.L);

}

ReduxC
ReduxC::fullReduction(cl::CommandQueue &q) {

  execute(q);
  ReduxC subredux(b_result, Nr, ctx, dev, ipt, libsplash, reducer);
  subredux.execute(q);
  return subredux;

}

