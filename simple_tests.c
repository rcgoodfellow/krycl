#include "krycl.h"
#include "krytest.h"
#include "mulSmDv.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
  
kryGPUInfo ginfo;

int setup()
{
  kryInit();
  int err = kryGetAGPU(&ginfo);
  return err;
}

KRY_TEST(mulSmDv)
{
  double  v[] = {4,7,2,7,9,5,2,6,5,3};
  cl_uint c[] = {0,1,2,0,1,3,0,2,1,3};
  cl_uint r[] = {0,3,6,8,10};
  krySparseMatrix A = {.v=v, .c=c, .r=r, .N=10, .n=4};
  double x0[] = {1,1,1,1};

  double *x1 = NULL;
  int err = kryMulSmDv(&ginfo, &A, x0, &x1, NULL);
  if(err) return -1;

  KRY_EXPECT_EQ(13, x1[0]);
  KRY_EXPECT_EQ(21, x1[1]);
  KRY_EXPECT_EQ(8,  x1[2]);
  KRY_EXPECT_EQ(8,  x1[3]);

  return KRY_SUCCESS;
}

int main()
{
  int err = setup();
  if(err) 
  {
    fprintf(stderr, "mulSmDv Test failed - see kry.log for details");
    return EXIT_FAILURE;
  }

  KRY_RUNTEST(mulSmDv);

  return EXIT_SUCCESS;
}
