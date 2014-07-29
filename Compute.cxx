#include "Compute.hxx"

using namespace splash;
  
Computation::Computation(REAL *x, size_t Ni, size_t ipt, 
    cl::Device dev, cl::Context ctx, cl::Program l)
  
  : x{x}, Ni{Ni}, ipt{ipt}, dev{dev}, 
    ctx{ctx}, libsplash{l}, resident_input{false} {}

Computation::Computation(cl::Buffer b_x, size_t Ni, size_t ipt, 
    cl::Device dev, cl::Context ctx, cl::Program l)
  
  : b_x{b_x}, Ni{Ni}, ipt{ipt}, dev{dev}, 
    ctx{ctx}, libsplash{l}, resident_input{true} {}

void Computation::readback(cl::CommandQueue &q) {

  q.enqueueReadBuffer(b_result, CL_TRUE, 0, sizeof(REAL)*Nr, result);

}
 
