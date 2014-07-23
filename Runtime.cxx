#include "Runtime.hxx"

using namespace splash;
using std::runtime_error;
using std::to_string;
using std::vector;

void PlatformGroup::resolveGPUs() {
  cl_int err = platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
  if(err != CL_SUCCESS) {
    throw runtime_error(
          "error fetching OpenCL devices for platform: " +
          platform.getInfo<CL_PLATFORM_NAME>()
      );
  }
}

std::vector<PlatformGroup> splash::resolvePlatformGroups()
{
  vector<PlatformGroup> pgroups;
  vector<cl::Platform> platforms;
  cl_int err = cl::Platform::get(&platforms);
  if(err != CL_SUCCESS) {
    throw runtime_error("error fetching OpenCL platforms: " + to_string(err));
  }
  if(platforms.empty()) {
    throw runtime_error("no OpenCL platforms found");
  }

  for(auto p : platforms) { 
    auto pg = PlatformGroup(p);

    //get the platform GPUs
    pg.resolveGPUs();
    if(pg.gpus.empty()) { continue; }

    //create a platform group context
    cl_int err{CL_SUCCESS};
    pg.ctx = cl::Context(pg.gpus, 
        nullptr, //no context props intially
        nullptr, //no error callbacks from ocl
        nullptr, //no user data
        &err);
    if(err != CL_SUCCESS) {
      throw runtime_error("error creating opencl context for platform " +
          pg.platform.getInfo<CL_PLATFORM_NAME>() + " " + to_string(err));
    }
    
    //create a platform group command queue
    for(size_t i=0; i<pg.gpus.size(); ++i) {
      pg.gqs.push_back(
          cl::CommandQueue(pg.ctx, 
            pg.gpus[i], 
            0, //no command queue props by default
            &err));
      if(err != CL_SUCCESS) {
        throw runtime_error("error creating opencl queue for gpu device in"
            " platform" + pg.platform.getInfo<CL_PLATFORM_NAME>() + " " +
            to_string(err));
      }
    }
      
    pgroups.push_back(pg);

  }

  return pgroups;
}

cl::Program*
PlatformGroup::loadProgram(cl::Program::Sources src, const char *build_opts) {
  cl_int err{CL_SUCCESS};
  cl::Program *p = new cl::Program(ctx, src, &err);
  if(err != CL_SUCCESS) {
    throw runtime_error("error creating program for platform " +
        platform.getInfo<CL_PLATFORM_NAME>() + " " + to_string(err));
  }
 
  err = p->build(build_opts);
  std::string build_log{};
  if(err != CL_SUCCESS) {
    for(auto &g : gpus) {
      build_log += p->getBuildInfo<CL_PROGRAM_BUILD_LOG>(g);
    }
    throw runtime_error("program build failure: " + build_log);
  }

  progs.push_back(p);
  return p;
}

cl::Kernel* PlatformGroup::loadKernel(cl::Program *p, std::string name) {
  cl_int err{CL_SUCCESS};
  cl::Kernel *k = new cl::Kernel(*p, name.c_str(), &err); 
  if(err != CL_SUCCESS) {
    throw runtime_error("kernel load failure for " + name + " " + to_string(err));
  }
  kernels[name] = k;
  return k;
}
  
cl::Buffer* 
PlatformGroup::loadBuffer(std::string name, cl_mem_flags mem_flags, 
    size_t data_size, void *host_ptr) {
  cl_int err{CL_SUCCESS};
  cl::Buffer *b = new cl::Buffer(ctx, mem_flags, data_size, host_ptr, &err);
  if(err != CL_SUCCESS) {
    throw runtime_error("buffer load failure for platform " +
        platform.getInfo<CL_PLATFORM_NAME>() + " " + to_string(err));
  }
  bufs[name] = b;
  return b;
}
