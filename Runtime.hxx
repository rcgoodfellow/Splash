#ifndef SPLASH_RUNTIME_HXX
#define SPLASH_RUNTIME_HXX


#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <utility>
#include <CL/cl.hpp>

namespace splash {

  
struct PlatformGroup {
  cl::Platform platform;
  cl::Context ctx;
  std::vector<cl::Device> gpus;
  std::vector<cl::CommandQueue> gqs;
  std::vector<cl::Program*> progs;
  std::unordered_map<std::string,cl::Kernel*> kernels;
  std::unordered_map<std::string,cl::Buffer*> bufs;

  PlatformGroup() = default;
  PlatformGroup(cl::Platform p) : platform{p} {}

  void resolveGPUs();
  cl::Program* loadProgram(cl::Program::Sources, const char *build_opts=nullptr);
  cl::Kernel* loadKernel(cl::Program *p, std::string name);
  cl::Buffer* loadBuffer(std::string name, cl_mem_flags mem_flags, 
      size_t data_sz, void *host_ptr);
};

std::vector<PlatformGroup> resolvePlatformGroups();

}

#endif
