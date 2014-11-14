#include "krycl.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>

int main()
{
  double  v[] = {4,7,2,7,9,5,2,6,5,3};
  cl_uint c[] = {0,1,2,0,1,3,0,2,1,3};
  cl_uint r[] = {0,3,6,8,10};
  krySparseMatrix A = {.v=v, .c=c, .r=r, .N=10, .n=4};
  double b[] = {2,3,4,5};
  double x0[] = {1,1,1,1};
  double x[] = {0,0,0,0};

  kryGPUInfo ginfo;
  int err = kryGetAGPU(&ginfo);
  if(err) 
  {
    fprintf(stderr, "kryError = %d\n", err);
    if(clError) fprintf(stderr, "clError = %d\n", clError);
    exit(EXIT_FAILURE);
  }

  kryExecInfo xinfo;
  err = kryArnoldi(&ginfo, &xinfo, &A, b, x0, x);
  if(err) 
  {
    fprintf(stderr, "kryError = %d\n", err);
    if(clError) fprintf(stderr, "clError = %d\n", clError);
  
    if(err == KRY_BUILD_PROGRAM_ERROR)
    {
      char *log;
      kryCLCSpew(&xinfo, &log);
      printf("%s\n", log);
      free(log);
    }

    exit(EXIT_FAILURE);
  }
  printf("Arnoldi finished successfully\n");

  return EXIT_SUCCESS;
}
