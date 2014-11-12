/******************************************************************************
 *
 */;
#include "krycl.h"
#include "krycl_internal.h"

cl_int clError = 0;


int kryGetAGPU(kryGPUInfo *ginfo)
{
  cl_uint found = 0;
  clError = clGetPlatformIDs(1, &ginfo->pid, &found);
  if(clError) return KRY_CL_PLATFORM_CREATE_ERROR;
  if(!found) return KRY_NO_AVAILABLE_PLATFORMS;

  found = 0;
  clError = clGetDeviceIDs(ginfo->pid, CL_DEVICE_TYPE_GPU, 1, &ginfo->did, &found);
  if(clError) return KRY_CL_GPU_CREATE_ERROR;
  if(!found) return KRY_NO_AVAILABLE_GPUS;

  ginfo->ctx = clCreateContext(NULL, 1, &ginfo->did, NULL, NULL, &clError);
  if(clError) return KRY_CL_CONTEXT_CREATE_ERROR;

  ginfo->q = clCreateCommandQueue(ginfo->ctx, ginfo->did, 0, &clError);
  if(clError) return KRY_CL_COMMAND_Q_CREATE_ERROR;

  return KRY_SUCCESS;
}

int _arnoldiAllocateMem(kryGPUInfo *ginfo, _arnoldiMem *amem, 
    krySparseMatrix *A, double *b, double *x0)
{
  amem->Av_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(double)*A->N, 
                   A->v, 
                   &clError);
  if(clError) return KRY_BAD_SM_VALUE_PTR;

  amem->Ac_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(unsigned)*A->N, 
                   A->c, 
                   &clError);
  if(clError) return KRY_BAD_SM_COLUMN_PTR;

  amem->Ar_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(unsigned)*A->n, 
                   A->r, 
                   &clError);
  if(clError) return KRY_BAD_SM_ROW_PTR;

  amem->b_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(double)*A->n, 
                   b, 
                   &clError);
  if(clError) return KRY_BAD_RHS_PTR;

  amem->x0_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(double)*A->n, 
                   x0, 
                   &clError);
  if(clError) return KRY_BAD_X0_PTR;

  amem->x_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_WRITE_ONLY, 
                   sizeof(double)*A->n, 
                   NULL, 
                   &clError);
  if(clError) return KRY_BAD_X_PTR;

  return KRY_SUCCESS;
}

int kryArnoldi(kryGPUInfo *ginfo, 
    kryExecInfo *xinfo,
    krySparseMatrix *A, double *b, double *x0, double *x)
{
  xinfo->ginfo = ginfo;
  _arnoldiMem amem; 
  int err = _arnoldiAllocateMem(ginfo, &amem, A, b, x0);
  if(err) return err;

  err = _arnoldiLoadCLProgram(ginfo, &xinfo->prog);
  if(err) return err;

  return KRY_SUCCESS;
}

int _readProgramSource(const char* fn, char **src, size_t *sz)
{
  FILE *f = fopen(fn, "r");
  if(!f) return KRY_CL_SOURCE_NOT_FOUND;

  fseek(f, 0, SEEK_END);
  long _sz = ftell(f);
  rewind(f);
  *src = (char*)malloc(_sz);
  *sz = fread(*src, sizeof(char), _sz, f);
  if(*sz != (size_t)_sz) return KRY_INCOMPLETE_SOURCE_READ;
  fclose(f);

  return EXIT_SUCCESS;
}

int _arnoldiLoadCLProgram(kryGPUInfo *ginfo, cl_program *prog)
{
  const char *fn = "arnoldi.cl";
  char *src = NULL;
  size_t sz; 
  int err = _readProgramSource(fn, &src, &sz);
  if(err) return err;

  *prog = clCreateProgramWithSource(ginfo->ctx, 1, (const char**)&src, &sz, 
      &clError);
  if(clError) return KRY_CREATE_PROGRAM_ERROR;

  clError = clBuildProgram(*prog, 1, &ginfo->did, NULL, NULL, NULL);
  if(clError) return KRY_BUILD_PROGRAM_ERROR;

  free(src);
  return EXIT_SUCCESS;
}

int kryCLCSpew(kryExecInfo *xinfo, char **log)
{
  size_t sz;
  clError = clGetProgramBuildInfo(xinfo->prog, xinfo->ginfo->did,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
  if(clError) return KRY_BUILD_LOG_ACCESS_ERROR;

  *log = (char*)malloc(sz);

  clError = clGetProgramBuildInfo(xinfo->prog, xinfo->ginfo->did,
      CL_PROGRAM_BUILD_LOG, sz, *log, NULL);
  if(clError) return KRY_BUILD_LOG_ACCESS_ERROR;

  return EXIT_SUCCESS;
}
