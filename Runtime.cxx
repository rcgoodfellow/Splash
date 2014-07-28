#include "Runtime.hxx"

using namespace splash;
using std::runtime_error;
using std::to_string;
using std::vector;
using std::string;
using std::make_pair;


void PlatformGroup::resolveGPUs() {
  platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
}

std::vector<PlatformGroup> splash::resolvePlatformGroups()
{
  cl_int err{CL_SUCCESS};
  vector<PlatformGroup> pgroups;
  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if(platforms.empty()) {
    throw runtime_error("no OpenCL platforms found");
  }

  for(auto p : platforms) { 
    auto pg = PlatformGroup(p);

    //get the platform GPUs
    pg.resolveGPUs();
    if(pg.gpus.empty()) { continue; }

    //create a platform group context
    pg.ctx = cl::Context(pg.gpus, 
        nullptr, //no context props intially
        nullptr, //no error callbacks from ocl
        nullptr, //no user data
        &err);
    if(err != CL_SUCCESS) {
      throw runtime_error{"failed to create context for platform " +
        pg.platform.getInfo<CL_PLATFORM_NAME>() + std::to_string(err) };
    }
    
    //create a platform group command queue
    for(size_t i=0; i<pg.gpus.size(); ++i) {
      pg.gqs.push_back(
          cl::CommandQueue(pg.ctx, 
            pg.gpus[i], 
            0, &err)); //no command queue props by default
      if(err != CL_SUCCESS) {
        throw runtime_error{"failed to create command queue for platform " +
          pg.platform.getInfo<CL_PLATFORM_NAME>() + std::to_string(err) };
      }}
    pgroups.push_back(pg);
  }
  return pgroups;
}

cl::Program* 
PlatformGroup::loadProgram(std::vector<std::string> filenames, 
    std::string build_opts) {
  cl::Program::Sources *src = new cl::Program::Sources;
  for(std::string fn : filenames) {
    std::string *s = new std::string(read_file(fn));
    src->push_back(std::make_pair(s->c_str(), s->length()));
  }
  return loadProgram(src, build_opts.c_str());
}

cl::Program*
PlatformGroup::loadProgram(cl::Program::Sources *src, std::string build_opts) {
  cl::Program *p = new cl::Program(ctx, *src);
  try {p->build(build_opts.c_str()); }
  catch(cl::Error &) {
    std::string build_log{};
    for(auto &g : gpus) {
      build_log += p->getBuildInfo<CL_PROGRAM_BUILD_LOG>(g);
    }
    throw runtime_error("exc program build failure: " + build_log);
  }
  sources.push_back(src);
  progs.push_back(p);
  return p;
}

cl::Kernel* PlatformGroup::loadKernel(cl::Program *p, std::string name) {
  cl::Kernel *k;
  try{k = new cl::Kernel(*p, name.c_str());}
  catch(cl::Error&) {
    throw runtime_error("exc: unkown kernel `" + name + "`");
  }
  kernels[name] = k;
  return k;
}
  
cl::Buffer* 
PlatformGroup::loadBuffer(std::string name, cl_mem_flags mem_flags, 
    size_t data_size, void *host_ptr) {
  cl::Buffer *b;
  try{b = new cl::Buffer(ctx, mem_flags, data_size, host_ptr);}
  catch(cl::Error&){
    throw runtime_error("error creating buffer for platform " 
        + platform.getInfo<CL_PLATFORM_NAME>());
  }
  bufs[name] = b;
  return b;
}

LibSplash::LibSplash(string splashdir)
  : splashdir{splashdir} {

    build_opts = "-I " + splashdir + " -DREAL=double";
    readSource();
}

void
LibSplash::readSource() {

  src_txt = read_file(splashdir + "Redux.cl");
  src = {
    make_pair(src_txt.c_str(), src_txt.length())
  };

}

cl::Program
LibSplash::get(cl::Context ctx) {

  cl::Program libsplash(ctx, src);
  try{ 
    
    libsplash.build(build_opts.c_str()); 
  
  }
  catch(cl::Error&) {
    throw runtime_error(
        libsplash.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
          ctx.getInfo<CL_CONTEXT_DEVICES>()[0]));
  }
  
  return libsplash;
}

GPUEnv::GPUEnv() {

  vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  platform = platforms[0];

  vector<cl::Device> gpus;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &gpus);
  dev = gpus[0];
  ctx = cl::Context(dev);
  q = cl::CommandQueue(ctx, dev);

}
