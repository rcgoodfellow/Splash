#ifndef SPLASH_THREADS_HXX
#define SPLASH_THREADS_HXX
/*****************************************************************************
 *  The Splash Project
 *
 *  runtime engine threading components - 25 July '14
 *  ~ ry
 *
 *  This file contains the threading components of the Splash engine
 */

#include <functional>
#include <utility>
#define __CL_ENABLE_EXCEPTOINS
#include <CL/cl.hpp>

namespace splash {

/*= DExecStrategy ============================================================
 *
 * A Device Execution Strategy is a function that computes the sizes of the 
 * thread execution ranges given the input data size and device capabilities. 
 * This type of function is used as a parameter for many hybrid compute functions.
 *
 * Input:
 *  @N - The global input size
 *  @dev - The OpenCL device for which the memory will be allocated
 * 
 * Returns:
 *  @p<@G,@L> - @p is a std::pair where the @G is the global execution range and 
 *              @L is the local execution range
 */

using DExecStrategy
  = std::function< std::pair<cl::NDRange, cl::NDRange>(size_t, cl::Device) >;

}

#endif
