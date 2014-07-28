#include "Compute.hxx"

using namespace splash;
  
Computation::Computation(cl::Device dev, cl::Context ctx, cl::Program l)
  : dev{dev}, ctx{ctx}, libsplash{l} {}

void Computation::readback(cl::CommandQueue &q) {

  q.enqueueReadBuffer(b_result, CL_TRUE, 0, sizeof(REAL)*Nr, result);

}
  
VectorC::VectorC(REAL *x, size_t N, size_t ipt,
    cl::Device dev, cl::Context ctx, cl::Program l)
  : Computation{dev, ctx, l}, x{x}, N{N}, ipt{ipt}, resident_input{false} { }
  
VectorC::VectorC(cl::Buffer b_x, size_t N, size_t ipt,
    cl::Device dev, cl::Context ctx, cl::Program l)
  : Computation{dev, ctx, l}, b_x{b_x}, N{N}, ipt{ipt}, resident_input{true} { }

