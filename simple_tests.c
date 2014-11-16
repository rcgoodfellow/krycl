#include "krycl.h"
#include "krytest.h"
#include "mulSmDv.h"
#include <stdbool.h>
  
kryGPUInfo ginfo;
FILE *results;

int setup()
{
  kryInit();
  int err = kryGetAGPU(&ginfo);
  results = fopen("simple_tests_results", "w+");
  return err;
}

void teardown()
{
  fclose(results);
}

bool chosen(cl_uint *begin, cl_uint *end, cl_uint v)
{
  for(cl_uint *i = begin; i != end; ++i)
  {
    if(*i == v) return true;
  }
  return false;
}

krySparseMatrix genSparseMatrix(cl_uint n, cl_uint rmin, cl_uint rmax)
{
  krySparseMatrix A;
  A.N = 0;
  A.n = n;
  A.r = (cl_uint *)malloc(sizeof(cl_uint)*(n+1));
  
  srand48(time(NULL));
  cl_uint ermax = rmax - rmin;
  A.r[0] = 0;
  for(cl_uint i=1; i<n+1; ++i)
  {
    cl_uint i_sz = rmin + lrand48()%ermax;
    A.r[i] = i_sz + A.r[i-1];
    A.N += i_sz;
  }

  A.v = (double *)malloc(sizeof(double)*A.N);
  A.c = (cl_uint *)malloc(sizeof(cl_uint)*A.N);

  for(cl_uint i=0; i<A.N; ++i)
  {
    A.v[i] = lrand48()%100 + drand48();
    A.c[i] = -1;
  }

  //diags
  for(cl_uint i=0; i<A.n; ++i)
  {
    A.c[A.r[i]] = i;
  }

  //off diags
  for(cl_uint i=0; i<A.n; ++i)
  {
    cl_uint rbegin = A.r[i],
            rend   = A.r[i+1];
    for(cl_uint j=rbegin+1; j<rend; ++j)
    {
      cl_uint z = lrand48()%n;
      while(chosen(&A.c[rbegin], &A.c[rend], z)) 
      {
        //printf("%u ", z);
        z = lrand48()%n;
      }
      A.c[j] = z;
    }
  }

  return A;
}

KRY_TEST(mulSmDv10)
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

KRY_TEST(mulSmDv10r)
{
  krySparseMatrix A = genSparseMatrix(10, 2, 6);
  fprintf(results, "%s\n", "mulSmDv10r matrix:");
  kryPrintSparseMatrix(results, A);
  //double x0[] = {1,1,1,1};
  //double *x1 = NULL;
  //int err = kryMulSmDv(&ginfo, &A, x0, &x1, NULL);
  //if(err) return -1;
  return KRY_SUCCESS;
}

int main()
{
  if(setup()) 
  {
    fprintf(stderr, "mulSmDv Test setup failed - see kry.log for details");
    return EXIT_FAILURE;
  }

  KRY_RUNTEST(mulSmDv10);
  KRY_RUNTEST(mulSmDv10r);

  teardown();

  return EXIT_SUCCESS;
}
