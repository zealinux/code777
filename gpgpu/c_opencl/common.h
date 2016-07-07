#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

char* read_source(const char*);
void random_fill(cl_float[], size_t);
