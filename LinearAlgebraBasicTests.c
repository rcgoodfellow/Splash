#include "SparseMatrix.h"
#include "Utility.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#include <CL/cl.h>

#define SPLASHDIR "/home/ry/Splash/"

int main()
{
  printf("%s", "Running basic linear algebra tests\n");

  SparseMatrix *M = create_EmptySparseMatrix(5, 4);
  DenseVector *v = create_DenseVector(5);
  DenseVector *Mv = create_DenseVector(5);

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

  dv_set(v, 0, 1.2);
  dv_set(v, 1, 3.4);
  dv_set(v, 2, 5.6);
  dv_set(v, 3, 6.7);
  dv_set(v, 4, 7.8);

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
  char *sm_source = read_file(SPLASHDIR "SparseMatrix.c");
  char *mv_source = read_file(SPLASHDIR "MatrixVector.cl");
  
  if(!mv_source || !sm_source){ exit(-1); }

  const char *sources[2] = {mv_source, sm_source};

  cl_program program = clCreateProgramWithSource(context, 2, 
		  sources, NULL, &err);
  if(err)
  {
    printf("Error while creating OpenCL program %d\n", err);
    exit(err);
  }
  else
  {
    printf("OpenCL program created\n");
  }

  const char* opts = "-I " SPLASHDIR;
  err = clBuildProgram(program, 1, &device, opts, NULL, NULL);
  if(err != CL_SUCCESS)
  {
    printf("Error while building OpenCL program %d\n", err);

 
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* program_log = (char*)malloc(log_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size+1, 
		    program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);

    exit(err);
  }
  else
  {
    printf("OpenCL program built\n");
  }

  //kernel creation
  cl_kernel kernel = clCreateKernel(program, "matrix_vector_mul", &err);
  if(err != CL_SUCCESS)
  {
    printf("Error while creating matrix_vector_mul kernel %d\n", err);
    exit(err);
  }
  else
  {
    printf("matrix_vector_mul kernel created\n");
  }

  //arg 2 buffer
  size_t arg_2_sz = sizeof(REAL) * M->N * M->n;
  cl_mem arg_2 = clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
      arg_2_sz,
      M->values,
      &err);
  if(err != CL_SUCCESS)
  {
    printf("Error creating matrix value argument buffer %d\n", err);
    exit(err);
  }
  else
  {
    printf("Created OpenCL buffer for matrix data\n");
  }

  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_2);
  if(err != CL_SUCCESS)
  {
    printf("Error setting matrix_vector_mul arg 2 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 2\n");
  }

  //arg 3 buffer
  size_t arg_3_sz = sizeof(unsigned int) * M->N;
  cl_mem arg_3 = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      arg_3_sz,
      M->row_sizes,
      &err);
  if(err != CL_SUCCESS)
  {
    printf("Error creating matrix row size argument buffer %d\n", err);
    exit(err);
  }
  else
  {
    printf("Created OpenCL buffer for matrix row sizes\n");
  }

  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &arg_3);
  if(err != CL_SUCCESS)
  {
    printf("Error setting matrix_vector_mul arg 3 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 3\n");
  }

  //arg 4 buffer
  size_t arg_4_sz = sizeof(unsigned int) * M->N * M->n;
  cl_mem arg_4 = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      arg_4_sz,
      M->indices,
      &err);
  if(err != CL_SUCCESS)
  {
    printf("Error creating matrix indices argument buffer %d\n", err);
    exit(err);
  }
  else
  {
    printf("Created OpenCL buffer for matrix indices\n");
  }

  err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &arg_4);
  if(err != CL_SUCCESS)
  {
    printf("Error setting matrix_vector_mul arg 4 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 4\n");
  }
 
  //arg 5 buffer
  size_t arg_5_sz = sizeof(REAL) * v->N;
  cl_mem arg_5 = clCreateBuffer(context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      arg_5_sz,
      v->values,
      &err);
  if(err != CL_SUCCESS)
  {
    printf("Error creating vector data buffer %d\n", err);
    exit(err);
  }
  else
  {
    printf("Created OpenCL buffer for vector data\n");
  }

  err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &arg_5);
  if(err != CL_SUCCESS)
  {
    printf("Error setting matrix_vector_mul arg 5 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 5\n");
  }

  //arg 6 buffer
  size_t arg_6_sz = sizeof(REAL) * v->N;
  cl_mem arg_6 = clCreateBuffer(context,
      CL_MEM_WRITE_ONLY,
      arg_6_sz,
      NULL,
      &err);
  if(err != CL_SUCCESS)
  {
    printf("Error creating target vector data buffer %d\n", err);
    exit(err);
  }
  else
  {
    printf("Created OpenCL buffer for target vector data\n");
  }

  err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &arg_6);
  if(err != CL_SUCCESS)
  {
    printf("Error settting matrix_vector_mul arg 6 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 6\n");
  }

  //arg 0
  err = clSetKernelArg(kernel, 0, sizeof(unsigned int), &M->n);
  if(err != CL_SUCCESS)
  {
    printf("Error setting matrix_vector_mul arg 0 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 0\n");
  }

  //arg 1
  err = clSetKernelArg(kernel, 1, sizeof(unsigned int), &M->N);
  if(err != CL_SUCCESS)
  {
    printf("Error settting matrix_vector_mul arg 1 %d\n", err);
    exit(err);
  }
  else
  {
    printf("Set matrix_vector_mul arg 1\n");
  }

  //execute kernel
  cl_uint work_dim = 1;
  size_t globalWorkSize[1] = { M->N };
  size_t localWorkSize[1] = { 1 };

  err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, globalWorkSize, 
      localWorkSize, 0, NULL, NULL);
  if(err != CL_SUCCESS)
  {
    printf("Failed to enqueue matrix_vector_mul kernel %d\n", err);
    exit(err);
  }
  else
  {
    printf("Sucessfully executed matrix_vector_mul kernel\n");
  }

  //get results
  err = clEnqueueReadBuffer(queue, arg_6,
      CL_TRUE, //blocking read
      0,
      sizeof(REAL) * Mv->N,
      Mv->values,
      0, //events in wait list
      NULL, //event wait list
      NULL //event
      );
  if(err != CL_SUCCESS)
  {
    printf("Failed to read results from matrix_vector_mul kernel %d\n", err);
    exit(err);
  }
  else
  {
    printf("Successfully read back matrix_vector_mul kernel results\n");
  }

  printf("Results *********************\n");
  for(unsigned int i=0; i<Mv->N; ++i)
  {
    printf("%f\n", Mv->values[i]);  
  }

  destroy_DenseVector(Mv);
  destroy_DenseVector(v);
  destroy_SparseMatrix(M);
  return EXIT_SUCCESS;
}
