#include "Redux.hxx"

using namespace splash;
using std::pair;

#include <iostream>

DAllocStrategy
splash::rdx_das_local_max(size_t elem_per_pe) {

  return 
    [elem_per_pe](size_t N, cl::Device dev) -> pair<size_t, size_t> {

      cl_ulong Nl = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / sizeof(REAL);
      std::cout << "local: " << Nl << std::endl;

      pair<cl::NDRange, cl::NDRange> exs = rdx_des_local_max(elem_per_pe)(N, dev);
      cl::NDRange G = exs.first;
      cl::NDRange L = exs.second;
      size_t groups = (G[0] / L[0]) * (G[1] / L[1]);

      Nl = L[0] * L[1];

      std::cout << "groups: " << groups << std::endl;
      cl_ulong Ng = groups;
      return {Nl, Ng};

    };

}

DExecStrategy
splash::rdx_des_local_max(size_t elem_per_pe) {
  
  return
    [elem_per_pe](size_t N, cl::Device dev) -> pair<cl::NDRange, cl::NDRange> {
 
      size_t 
        max_local = dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
        L0 = sqrt(max_local),
        L1 = L0;

      size_t total_thds = ceil(N / (float)elem_per_pe);
      size_t
        G0 = sqrt(total_thds),
        G1 = G0;
      G0 += (L0 - G0 % L0);
      G1 += (L1 - G1 % L1);
      
      return {cl::NDRange{G0, G1}, cl::NDRange{L0, L1}};

    };

}


ReduxMem
splash::redux_alloc(REAL *x, size_t N, cl::Context ctx, cl::Device dev, 
    DAllocStrategy das) {

  ReduxMem m;
  m.x = x;
  m.N = N;
  pair<int,int> p = das(N, dev);
  m.Nl = p.first;
  m.Ng = p.second;
  m.ipt = 64;

  m.b_x = cl::Buffer(
      ctx, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
      sizeof(REAL) * N, 
      x);

  std::cout << "G: (bytes) " << sizeof(REAL) * m.Ng << std::endl;
  m.gs = (REAL*)malloc(sizeof(REAL)*m.Ng);
  m.grspace = cl::Buffer(
      ctx,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(REAL) * m.Ng,
      m.gs);

  return m;
}

cl::Kernel
splash::redux_add_kernel(cl::Program splash, ReduxMem m) {

  cl::Kernel k = cl::Kernel(splash, "redux_add");
  k.setArg(0, m.b_x);
  k.setArg(1, m.N);
  k.setArg(2, m.ipt);
  k.setArg(3, cl::Local(m.Nl * sizeof(REAL)));
  k.setArg(4, m.grspace);

  return k;

}
