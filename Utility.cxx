#include "Utility.hxx"
#include <stdio.h>
#include <fstream>
#include <sstream>

std::string read_file(std::string filename)
{
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();

  return std::string(buffer.str());
}
