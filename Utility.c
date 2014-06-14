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
  fseek(f, 0L, SEEK_SET);

  char *data = (char*)malloc(sz + 1);
  fread(data, 1, sz, f);
  fclose(f);

  //Null terminate
  data[sz] = '\0';

  return data;
}
