#include "Utility.h"
#include <stdio.h>

char* read_file(const char* filename)
{
  FILE *f = fopen(filename, "r");

  if(!f)
  {
    printf("Unable to read file: %s\n", filename);
    return NULL;
  }
  fseek(f, 0L, SEEK_END);
  long sz = ftell(f);
  rewind(f);

  char *data = (char*)malloc(sz);
  fread(data, 1, sz, f);

  return data;
}
