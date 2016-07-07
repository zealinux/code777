#include "common.h"

#include <stdio.h>

// 其他函数
char* read_source(const char* filename) {
  FILE *h = fopen(filename, "r");
  fseek(h, 0, SEEK_END);
  size_t s = ftell(h);
  rewind(h);
  char* program = (char*)malloc(s + 1);
  fread(program, sizeof(char), s, h);
  program[s] = '\0';
  fclose(h);
  return program;
}

void random_fill(cl_float array[], size_t size) {
  for (int i = 0; i < size; ++i)
    array[i] = (cl_float)rand() / RAND_MAX;
}
