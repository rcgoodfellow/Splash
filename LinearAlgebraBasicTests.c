#include "SparseMatrix.h"
#include "Utility.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <OpenCL/cl.h>

int main()
{
  printf("%s", "Running basic linear algebra tests\n");

  SparseMatrix *M = create_EmptySparseMatrix(5, 4);
  DenseVector *v = create_DenseVector(5);

  sm_set(M, 0, 0, 4);
  sm_set(M, 0, 1, 4);

  sm_set(M, 1, 0, 4);
  sm_set(M, 1, 3, 3);
  sm_set(M, 1, 2, 2);

  sm_set(M, 2, 3, 5);
  sm_set(M, 2, 2, 7);
  sm_set(M, 2, 1, 2);
 
  sm_set(M, 3, 2, 5);
  sm_set(M, 3, 1, 3);
  sm_set(M, 3, 4, 7);
  sm_set(M, 3, 3, 15);

  sm_set(M, 4, 3, 7);
  sm_set(M, 4, 4, 7);

  //sm_print(M);

  //platform
  cl_platform_id platform;
  cl_uint n_platforms = 0;
  cl_int err = clGetPlatformIDs(1, &platform, &n_platforms);
  if(err != CL_SUCCESS)
  {
    printf("%s %d", "Error while getting OpenCL platform\n", err);
    exit(err);
  }
  if(n_platforms == 0)
  {
    printf("%s", "Error: no OpenCL platforms on this system\n");
    exit(-1);
  }
  else
  {
    printf("%u %s", n_platforms, "OpenCL platforms found\n");
  }

  //gpu device
  cl_device_id device;
  cl_uint n_gpu;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &n_gpu);
  if(err != CL_SUCCESS)
  {
    printf("%s %d", "Error while getting GPU devices\n", err);
    exit(err);
  }
  if(n_gpu == 0) 
  {
    printf("%s", "Error: no GPU devices available\n");
    exit(-1);
  }
  else
  {
    printf("%u %s", n_gpu, "GPU devices found\n");
  }

  //context
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if(err != CL_SUCCESS)
  {
    printf("Error while creating OpenCL context %d\n", err);
    exit(err);
  }
  else
  {
    printf("OpenCL context created\n");
  }

  //command queue
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  if(err != CL_SUCCESS)
  {
    printf("Error while creating OpenCL command queue %d\n", err);
    exit(err);
  }
  else
  {
    printf("OpenCL command queue created\n");
  }

  //program compilation
  char *source = read_file("MatrixVector.cl");
  if(!source){ exit(-1); }

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
  if(err)
  {
    printf("Error while creating OpenCL program %d\n", err);
    exit(err);
  }
  else
  {
    printf("OpenCL program created\n");
  }

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if(err != CL_SUCCESS)
  {
    printf("Error while building OpenCL program %d\n", err);

 
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* program_log = (char*)malloc(log_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);

    exit(err);
  }

  free(source);
  destroy_DenseVector(v);
  destroy_SparseMatrix(M);
  return EXIT_SUCCESS;
}
