#include "Redux.hxx"

using namespace splash;
using std::pair;

#include <iostream>

void
ReduxC::computeMemoryRequirements() {

    //The global memory is the nuber of required workgroups becuase each
    //workgroup eventually writes its result back to a point in the global 
    //array. Since the redux kernel is a 2 dimensional kernel, the total 
    //number of workgroups is computed as the product of the number of 
    //workgroups in each dimension.
    size_t gmem = (G[0] / L[0]) * (G[1] / L[1]);

    //The local memory is computed as the size of the workgroup. Each
    //thread within a workgroup reduces @elem_per_pe elements onto a single
    //point in the LDS, thus the LDS must be the size of the workgroup
    size_t lmem = L[0] * L[1];

    Nl = lmem;
    Ng = gmem;
}

void
ReduxC::computeThreadStrategy() {
  
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
    
    G = cl::NDRange{G0, G1};
    L = cl::NDRange{L0, L1};

}


ReduxC::ReduxC(REAL *x, size_t N, cl::Context ctx, cl::Device dev, 
    size_t ipt, cl::Program splashp)
  : x{x}, N{N}, ipt{ipt}, dev{dev} {

    computeThreadStrategy();
    computeMemoryRequirements();

  b_x = cl::Buffer(
      ctx, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
      sizeof(REAL) * N, 
      x);

  gs = (REAL*)malloc(sizeof(REAL)*Ng);
  grspace = cl::Buffer(
      ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(REAL) * Ng,
      gs);
  
  k = cl::Kernel(splashp, "redux_add");

  k.setArg(0, b_x);
  k.setArg(1, N);
  k.setArg(2, ipt);
  k.setArg(3, cl::Local(Nl * sizeof(REAL)));
  k.setArg(4, grspace);

}

REAL
ReduxC::execute(cl::CommandQueue &q) {

  q.enqueueNDRangeKernel(k, cl::NullRange, G, L);
  q.enqueueReadBuffer(grspace, CL_TRUE, 0, sizeof(REAL)*Ng, gs);
  REAL r{0};
  for(size_t i=0; i<Ng; ++i) { r += gs[i]; }
  return r;

}

