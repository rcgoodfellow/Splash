#ifndef SPLASH_MEMORY_HXX
#define SPLASH_MEMORY_HXX
/******************************************************************************
 *  The Splash Project
 *
 *  runtime engine memory components - 25 July '14
 *  ~ ry
 *
 *  This file contains the memory components of the Splash engine
 */

#include <functional>
#include <utility>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace splash {

/*= DAllocStrategy ============================================================
 *
 * A Device Allocation Strategy is a function that computes the sizes of local
 * and global memory spaces for use on devices given an input data size. And
 * device capabilities This type of function is used as a parameter for many 
 * hybrid compute functions
 *
 * Input:
 *  @N - The global input size
 *  @dev - The OpenCL device for which the memory will be allocated
 * 
 * Returns:
 *  @p<@Nl,@Ng> - @p is a std::pair where the @Nl is the local allocation size per
 *                compute unit and @Ng is the global allocation size
 */
using DAllocStrategy 
  = std::function< std::pair<size_t, size_t>(size_t, cl::Device) >;

}

#endif
