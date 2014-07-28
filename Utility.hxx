#ifndef __SPLASH_UTILITY_
#define __SPLASH_UTILITY_

#include "API.h"

#include <stdlib.h>
#include <string>
#include <stdexcept>
#include <random>

namespace splash {

std::string read_file(std::string filename);

REAL* random_vector(size_t sz);

}

#endif
