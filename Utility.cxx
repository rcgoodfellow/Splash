#include "Utility.hxx"
#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace splash;
using namespace std;

std::string splash::read_file(std::string filename) {
  std::ifstream t(filename);
  if(!t.good()) {
    t.close();
    throw runtime_error("Unable to read file: " + filename);
  }
  std::stringstream buffer;
  buffer << t.rdbuf();

  return std::string(buffer.str());
}

REAL* splash::random_vector(size_t sz) {
  random_device rd;
  uniform_real_distribution<REAL> i_dst{0.5, 10};
  normal_distribution<REAL> v_dst;
  default_random_engine re{rd()};
  
  v_dst = normal_distribution<REAL>{i_dst(re), 15};
  REAL *x = (REAL*)malloc(sizeof(REAL)*sz);
  for(unsigned int i=0; i<sz; ++i) { x[i] = v_dst(re); }
  return x;

}
