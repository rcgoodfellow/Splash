#include "Runtime.hxx"
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;
using namespace splash;

int main() {
  PlatformGroup pg = resolvePlatformGroups()[0];

  size_t sz1k = sizeof(double)*1e3,
         sz10k = sizeof(double)*1e4,
         sz100k = sizeof(double)*1e5,
         sz1M = sizeof(double)*1e6,
         sz10M = sizeof(double)*1e7,
         sz100M = sizeof(double)*1e8;

  double *d1k = (double*)malloc(sz1k),
         *d10k = (double*)malloc(sz10k), 
         *d100k = (double*)malloc(sz100k),
         *d1M = (double*)malloc(sz1M), 
         *d10M = (double*)malloc(sz10M), 
         *d100M = (double*)malloc(sz100M);

  cl_mem_flags MF = CL_MEM_COPY_HOST_PTR;
  cl::Buffer 
    *b_1k = pg.loadBuffer("1k", MF, sz1k, d1k),
    *b_10k = pg.loadBuffer("10k", MF, sz10k, d10k),
    *b_100k = pg.loadBuffer("100k", MF, sz100k, d100k),
    *b_1M = pg.loadBuffer("1M", MF, sz1M, d1M),
    *b_10M = pg.loadBuffer("10M", MF, sz10M, d10M),
    *b_100M = pg.loadBuffer("100M", MF, sz100M, d100M);
    
  cout << "Measurement resolution: " <<
    duration_cast<microseconds>(high_resolution_clock::duration(1)).count()
    << " us" << endl;

  cout << "measuring data transfer time " << endl;;
  

  cout << "1k" << endl;
  auto start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_1k, CL_TRUE, 0, sz1k, d1k);
  auto end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;

  cout << "10k" << endl;
  start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_10k, CL_TRUE, 0, sz10k, d10k);
  end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;

  cout << "100k" << endl;
  start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_100k, CL_TRUE, 0, sz100k, d100k);
  end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;

  cout << "1M" << endl;
  start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_1M, CL_TRUE, 0, sz1M, d1M);
  end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;

  cout << "10M" << endl;
  start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_10M, CL_TRUE, 0, sz10M, d10M);
  end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;

  cout << "100M" << endl;
  start = high_resolution_clock::now();
  pg.gqs[0].enqueueWriteBuffer(*b_100M, CL_TRUE, 0, sz100M, d100M);
  end  = high_resolution_clock::now();
  cout << duration_cast<microseconds>(end-start).count() << " us, ";
  cout << duration_cast<milliseconds>(end-start).count() << " ms" << endl;


  return EXIT_SUCCESS;
}
