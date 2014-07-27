#ifndef SPLASH_RUNTIME_HXX
#define SPLASH_RUNTIME_HXX
/******************************************************************************
 *  The Splash Project
 *
 *  reduction operations - 20 July '14
 *  ~ ry
 *
 *  This file contains host code for interacting with the OpenCL runtime
 */

#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "Utility.hxx"

namespace splash {
  
struct PlatformGroup {
  cl::Platform platform;
  cl::Context ctx;
  std::vector<cl::Device> gpus;
  std::vector<cl::CommandQueue> gqs;
  std::vector<cl::Program*> progs;
  std::unordered_map<std::string,cl::Kernel*> kernels;
  std::unordered_map<std::string,cl::Buffer*> bufs;
  std::vector<cl::Program::Sources*> sources;

  PlatformGroup() = default;
  PlatformGroup(cl::Platform p) : platform{p} {}

  void resolveGPUs();
  cl::Program* loadProgram(std::vector<std::string> filenames, 
      std::string build_opts);
  cl::Program* loadProgram(cl::Program::Sources*, std::string);
  cl::Kernel* loadKernel(cl::Program *p, std::string name);
  cl::Buffer* loadBuffer(std::string name, cl_mem_flags mem_flags, 
      size_t data_sz, void *host_ptr);
};

std::vector<PlatformGroup> resolvePlatformGroups();

struct LibSplash {

  cl::Program::Sources src;
  std::string src_txt, splashdir, build_opts;

  LibSplash(std::string splashdir);

  cl::Program get(cl::Context ctx);

  private:
    void readSource();

};

}

#endif
