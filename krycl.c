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

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
      sizeof(size_t)*3, ginfo->max_work_item_sizes, NULL);
  if(clError) return KRY_CL_DEVINFO_QUERY_ERROR;

  clError = clGetDeviceInfo(ginfo->did, CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(cl_uint), &ginfo->max_compute_units, NULL);
  if(clError) return KRY_CL_DEVINFO_QUERY_ERROR;

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
                   sizeof(cl_uint)*A->N, 
                   A->c, 
                   &clError);
  if(clError) return KRY_BAD_SM_COLUMN_PTR;

  amem->Ar_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(cl_uint)*(A->n + 1), 
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

  amem->r0_mem = 
    clCreateBuffer(ginfo->ctx,
                   CL_MEM_READ_WRITE,
                   sizeof(double)*A->n,
                   NULL,
                   &clError);
  if(clError) return KRY_BAD_R0_PTR;

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

  err = _arnoldiLoadCLProgram(ginfo, xinfo);
  if(err) return err;

  err = _arnoldiLoadKernels(xinfo, &amem, A->N, A->n);
  if(err) return err;

  err = _arnoldiExecute(ginfo, xinfo, A, &amem);
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

  return KRY_SUCCESS;
}

int _arnoldiLoadCLProgram(kryGPUInfo *ginfo, kryExecInfo *xinfo)
{
  const char *fn = "arnoldi.cl";
  char *src = NULL;
  size_t sz; 
  int err = _readProgramSource(fn, &src, &sz);
  if(err) return err;

  xinfo->prog = clCreateProgramWithSource(ginfo->ctx, 1, (const char**)&src, &sz, 
      &clError);
  if(clError) return KRY_CREATE_PROGRAM_ERROR;

  clError = clBuildProgram(xinfo->prog, 1, &ginfo->did, NULL, NULL, NULL);
  if(clError) return KRY_BUILD_PROGRAM_ERROR;

  free(src);
  return KRY_SUCCESS;
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

  return KRY_SUCCESS;
}

int _arnoldiLoadKernels(kryExecInfo *xinfo, _arnoldiMem *amem, cl_uint N,
    cl_uint n)
{
  xinfo->kernels = (cl_kernel*)malloc(sizeof(cl_kernel)*1); 
  xinfo->kernels[0] = clCreateKernel(xinfo->prog, "mul_sp_dv", &clError);
  if(clError) return KRY_CREATE_CL_KERNEL_ERROR;

  clError = clSetKernelArg(xinfo->kernels[0], 0, sizeof(cl_mem), &amem->Av_mem);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;

  clError = clSetKernelArg(xinfo->kernels[0], 1, sizeof(cl_mem), &amem->Ac_mem);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;

  clError = clSetKernelArg(xinfo->kernels[0], 2, sizeof(cl_mem), &amem->Ar_mem);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;
  
  clError = clSetKernelArg(xinfo->kernels[0], 3, sizeof(cl_mem), &amem->x0_mem);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;
  
  clError = clSetKernelArg(xinfo->kernels[0], 4, sizeof(cl_mem), &amem->r0_mem);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;

  clError = clSetKernelArg(xinfo->kernels[0], 5, sizeof(cl_uint), &n);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;

  clError = clSetKernelArg(xinfo->kernels[0], 6, sizeof(cl_uint), &N);
  if(clError) return KRY_SET_KERNEL_ARG_ERROR;

  return KRY_SUCCESS;
}

int _arnoldiExecute(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A,
    _arnoldiMem *amem)
{
  int err = _exec_mul_sp_dv(ginfo, xinfo, A);
  if(err) return err;

  double *Av;
  err = _getResult_mul_sp_dv(ginfo, amem, A, &Av);

  for(size_t i=0; i<(A->n-1); ++i)
  {
    printf("%f,", Av[i]);
  }
  printf("%f\n", Av[A->n-1]);

  return KRY_SUCCESS;
}

int _getResult_mul_sp_dv(kryGPUInfo *ginfo, _arnoldiMem *amem, 
    krySparseMatrix *A, double **Av)
{
  *Av = (double*)malloc(sizeof(double)*A->n);
  clError = clEnqueueReadBuffer(
      ginfo->q,
      amem->r0_mem,
      CL_TRUE,
      0,
      sizeof(double)*A->n,
      *Av,
      0,NULL,NULL);
  if(clError) return KRY_KERNEL_READBACK_ERROR;

  return KRY_SUCCESS;
}

int _exec_mul_sp_dv(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A)
{
  size_t global_sz, local_sz;
  int err = _getShape_mul_sp_dv(ginfo, A, &global_sz, &local_sz);
  if(err) return err;
  printf("[kexec] mul_sp_dv g=%zu l=%zu\n", global_sz, local_sz);
  clError = clEnqueueNDRangeKernel(
      ginfo->q,
      xinfo->kernels[0],
      1,
      0,
      &global_sz,
      &local_sz,
      0, NULL, NULL);
  if(clError) return KRY_EXEC_KERNEL_ERROR;

  return KRY_SUCCESS;
}

int _getShape_mul_sp_dv(kryGPUInfo *ginfo, krySparseMatrix *A, 
    size_t *gsz, size_t *lsz)
{
  size_t d0sz = ginfo->max_work_item_sizes[0];
  *lsz = A->n >= d0sz ? d0sz : A->n;
  //*gsz = ceil((double)A->n/(*lsz)); 
  *gsz = A->n;
  return KRY_SUCCESS;
}
