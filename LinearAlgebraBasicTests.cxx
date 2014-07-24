#include "Runtime.hxx"
#include "SparseMatrix.hxx"
#include <iostream>
#include <chrono>

using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::string;
using namespace splash;
using namespace std::chrono;

#define SPLASHDIR "/home/ry/Splash/"

vector<PlatformGroup> pgroups;
SparseMatrix *M;
DenseVector *v;
DenseVector *Mv; 

void initOclEnv() {
  pgroups = resolvePlatformGroups();

  size_t gpu_count{0};
  for(auto &pg : pgroups) { gpu_count += pg.gpus.size(); }

  cout << "Found " << pgroups.size() << " GPU platforms with "
    << gpu_count << " total GPUs" << endl;
}

void loadPrograms() {
  vector<string> srcs{
    SPLASHDIR "SparseMatrix.c",
    SPLASHDIR "MatrixVector.cl"};

  string build_opt = string("-I ") + SPLASHDIR;
  for(PlatformGroup &pg : pgroups) { 
    cl::Program *prog = pg.loadProgram(srcs, build_opt); 
    pg.loadKernel(prog, "matrix_vector_mul");
  } 
}

void loadData() {
  for(auto &pg : pgroups) {
    cl_mem_flags RC = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    //the matrix itself
    pg.loadBuffer("M", RC, sizeof(REAL) * M->N * M->n, M->values);
    //row sizes
    pg.loadBuffer("M_ri", RC, sizeof(unsigned int) * M->N, M->row_sizes);
    //indices
    pg.loadBuffer("M_idx", RC, sizeof(unsigned int) * M->N * M->n, M->indices);
    //vector values
    pg.loadBuffer("v", RC, sizeof(REAL) * v->N, v->values);
    //result vector
    pg.loadBuffer("Mv", CL_MEM_WRITE_ONLY, sizeof(REAL) * v->N, nullptr);
  }
}

void setKernelArgs() {
  for(auto &pg : pgroups) {
    cl::Kernel *k = pg.kernels.at("matrix_vector_mul");
    k->setArg(0, M->n);
    k->setArg(1, M->N);


    cout << "Measurement resolution: " <<
      duration_cast<nanoseconds>(high_resolution_clock::duration(1)).count()
      << " ns" << endl;

    cout << "measuring sparse matrix data transfer time...   ";
    auto start = high_resolution_clock::now();

    k->setArg(2, *pg.bufs.at("M"));

    auto end  = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(end-start);
    cout << dt.count() << " ns, "
         << duration_cast<microseconds>(dt).count() << " us"
         << endl;


    k->setArg(3, *pg.bufs.at("M_ri"));
    k->setArg(4, *pg.bufs.at("M_idx"));
    k->setArg(5, *pg.bufs.at("v"));
    k->setArg(6, *pg.bufs.at("Mv"));
  }
}

void enqueueKernels() {
  for(auto &pg : pgroups) {
    cl::Kernel *k = pg.kernels.at("matrix_vector_mul");
    for(auto &q : pg.gqs) {
      q.enqueueNDRangeKernel(*k, cl::NullRange, cl::NDRange(M->N), 
          cl::NDRange(1));
    }}
}

void collectResults() {
  for(auto &pg : pgroups) { for(auto &q : pg.gqs) {
    q.enqueueReadBuffer(*pg.bufs.at("Mv"), CL_TRUE, 0, sizeof(REAL) * Mv->N,
        Mv->values);
  }}
}

void printResults() {
  printf("Results *********************\n");
  for(unsigned int i=0; i<Mv->N; ++i) { printf("%f\n", Mv->values[i]); }
}

void buildMatricies() {
  M = create_EmptySparseMatrix(5, 4);
  v = create_DenseVector(5);
  Mv = create_DenseVector(5);

  smSetRow(M, 0, {{0,4},{1,4}});
  smSetRow(M, 1, {{0,4},{3,3},{2,3}});
  smSetRow(M, 2, {{3,5},{2,7},{1,2}});
  smSetRow(M, 3, {{2,5},{1,3},{4,7},{3,15}});
  smSetRow(M, 4, {{3,7},{4,7}});
  dvSet(v, {{0,1.2},{1,3.4},{2,5.6},{3,6.7},{4,7.8}});
}

int main()
{
  printf("%s", "Running basic linear algebra tests\n");

  buildMatricies();
  initOclEnv();
  loadPrograms();
  loadData();
  setKernelArgs();
  enqueueKernels();
  collectResults();
  printResults();

  return EXIT_SUCCESS;
}
