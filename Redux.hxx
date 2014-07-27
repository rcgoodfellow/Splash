#ifndef SPLASH_REDUX_HXX
#define SPLASH_REDUX_HXX
/******************************************************************************
 *  The Splash Project
 *
 *  reduction operations - 25 July '14
 *  ~ ry
 *
 *  This file contains host code for performing reduction operations on real
 *  numbers
 */

#include "API.h"
#include "Engine.hxx"
#include <cmath>

namespace splash {

/*= das_local_max =============================================================
*
* A device allocation strategy which maximizes the use of compute device
* local memory. For complete synopsis see DAllocStrategy documentation
*/

struct ReduxMem {
  REAL *x, *ls, *gs;
  cl::Buffer b_x, grspace;
  size_t N, Nl, Ng, ipt;
};

DAllocStrategy
rdx_das_local_max(size_t elem_per_pe=64);

DExecStrategy
rdx_des_local_max(size_t elem_per_pe=64);

ReduxMem
redux_alloc(REAL *x, size_t N, cl::Context ctx, cl::Device dev, DAllocStrategy das = rdx_das_local_max());

cl::Kernel
redux_add_kernel(cl::Program splash, ReduxMem m);

}
#endif
