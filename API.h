#ifndef SPLASH_API_H
#define SPLASH_API_H

#ifndef __OPENCL_VERSION__
  #include <CL/cl.h> 
  #ifdef __cplusplus
  #define API extern "C"
  #else
  #define API
  #endif
  #define __CL_ENABLE_EXCEPTIONS
  typedef cl_double REAL;
#else
  typedef double REAL;
  #define API
//  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


#ifndef __OPENCL_VERSION__
#else
#endif

#endif
