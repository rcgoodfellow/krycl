#include "core/krycl.h"
#include "test/krytest.h"
#include "core/mulSmDv.h"
#include <stdbool.h>
#include <mkl.h>
#include <string.h>
  
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

krySparseMatrix genSparseMatrix(cl_uint n, cl_uint rmin, cl_uint rmax, cl_uint vmax)
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
    A.v[i] = lrand48()%vmax + drand48();
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

double * randomDv(cl_uint sz, cl_uint vmax)
{
  double *v = (double *)malloc(sizeof(double)*sz);
  srand48(time(NULL));

  for(cl_uint i=0; i<sz; ++i) v[i] = lrand48()%(vmax-1) + drand48();

  return v;
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
  cl_uint n = 10,
          rmin = 2, 
          rmax = 6, 
          vmax = 10;

  krySparseMatrix A = genSparseMatrix(n, rmin, rmax, vmax);
  fprintf(results, "%s\n", "mulSmDv10r matrix:");
  kryPrintSparseMatrix(results, A);
  double *x0 = randomDv(n, vmax);

  double *x1 = NULL;
  int err = kryMulSmDv(&ginfo, &A, x0, &x1, NULL);
  if(err) return -1;
  return KRY_SUCCESS;
}

KRY_TEST(mulSmDv100000r)
{
  cl_uint n = 100000,
          rmin = 2, 
          rmax = 17, 
          vmax = 10;

  krySparseMatrix A = genSparseMatrix(n, rmin, rmax, vmax);
  double *x0 = randomDv(n, vmax);

  double *x1_mkl = (double *)malloc(sizeof(double)*A.n);
  MKL_INT *rb = (MKL_INT *)malloc(sizeof(MKL_INT)*A.n),
          *re = (MKL_INT *)malloc(sizeof(MKL_INT)*A.n);

  //for(cl_uint i=0; i<A.n; i++){ x1_mkl[i] = 0.0; }
  for(cl_uint i=0; i<n; i++)
  {
    rb[i] = (MKL_INT)A.r[i];
    re[i] = (MKL_INT)(A.r[i+1]);
  }

  double one = 1;
  struct timespec mkl0, mkl1,
                  kry0, kry1,
                  res;
  clock_getres(CLOCK_MONOTONIC_RAW, &res);
 
  clock_gettime(CLOCK_MONOTONIC_RAW, &mkl0); 
  for(int i=0; i<100; ++i)
  {
    for(cl_uint j=0; j<A.n; j++){ x1_mkl[j] = 0.0; }
    mkl_dcsrmv("N", (MKL_INT*)&n, (MKL_INT*)&n, &one, "G**C", A.v, (MKL_INT*)A.c,
       rb, re, x0, &one, x1_mkl);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &mkl1); 
  double mkltime = clockdiff(mkl0, mkl1, res);
  
  double *x1 = NULL;
  clock_gettime(CLOCK_MONOTONIC_RAW, &kry0); 
  kryExecInfo xinfo;
  int err = kryMulSmDv(&ginfo, &A, x0, &x1, &xinfo);
  for(int i=0; i<99; ++i)
  {
    kryKexec_mulSmDv(&ginfo, &xinfo, &A);
  }
  clWaitForEvents(1, &xinfo.kcomplete);
  clock_gettime(CLOCK_MONOTONIC_RAW, &kry1); 
  double krytime = clockdiff(kry0, kry1, res);


  for(cl_uint i=0; i<A.n; ++i)
  {
    KRY_EXPECT_DOUBLE_EQ(x1_mkl[i], x1[i], 1.0e-6);
  }

  free(x1_mkl);
  free(rb);
  free(re);

  free(x0);
  free(x1);
  free(A.v);
  free(A.r);
  free(A.c);
  
  fprintf(results, "mkl: %f\n", mkltime);
  fprintf(results, "kry: %f\n", krytime);

  if(err) return -1;
  return KRY_SUCCESS;
}

int main()
{
  if(setup()) 
  {
    fprintf(stderr, "mulSmDv Test setup failed - see kry.log for details\n");
    return EXIT_FAILURE;
  }

  //KRY_RUNTEST(mulSmDv10);
  //KRY_RUNTEST(mulSmDv10r);
  KRY_RUNTEST(mulSmDv100000r);
  //KRY_RUNTEST(mklBarf);

  teardown();

  return EXIT_SUCCESS;
}
