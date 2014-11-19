#include "core/mulSmDv.h"

int kryKalloc_mulSmDv(kryGPUInfo *ginfo, kryKdata_mulSmDv *kdata, krySparseMatrix *A, 
    double *v)
{
  kdata->A_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(double)*A->N, 
                   A->v, 
                   &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, 
        "%s: error allocating memory for sparse matrix values - clError %d", 
        __func__, clError);
    return KRY_KALLOC_ERROR;
  }

  kdata->c_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(cl_uint)*A->N, 
                   A->c, 
                   &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, 
        "%s: error allocating memory for sparse matrix columns - clError %d", 
        __func__, clError);
    return KRY_KALLOC_ERROR;
  }

  kdata->r_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(cl_uint)*(A->n + 1), 
                   A->r, 
                   &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, 
        "%s: error allocating memory for sparse matrix rows - clError %d", 
        __func__, clError);
    return KRY_KALLOC_ERROR;
  }

  kdata->v_mem = 
    clCreateBuffer(ginfo->ctx, 
                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                   sizeof(double)*A->n, 
                   v, 
                   &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, 
        "%s: error allocating memory for dense vector - clError %d", 
        __func__, clError);
    return KRY_KALLOC_ERROR;
  }

  kdata->Av_mem = 
    clCreateBuffer(ginfo->ctx,
                   CL_MEM_WRITE_ONLY,
                   sizeof(double)*A->n,
                   NULL,
                   &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL, 
        "%s: error allocating memory for dense vector - clError %d", 
        __func__, clError);
    return KRY_KALLOC_ERROR;
  }

  return KRY_SUCCESS;
}

int kryKload_mulSmDv(kryExecInfo *xinfo, kryKdata_mulSmDv *kdata, cl_uint N, 
    cl_uint n)
{
  xinfo->kernels = (cl_kernel*)malloc(sizeof(cl_kernel)*1); 
  xinfo->kernels[0] = clCreateKernel(xinfo->ginfo->kry_core, "mulSmDv", &clError);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error loading the mulSmDv kernel from krycore - clError %d",
        __func__, clError);
    return KRY_CREATE_CL_KERNEL_ERROR;
  }

  clError = clSetKernelArg(xinfo->kernels[0], 0, sizeof(cl_mem), &kdata->A_mem);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 0 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }

  clError = clSetKernelArg(xinfo->kernels[0], 1, sizeof(cl_mem), &kdata->c_mem);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 1 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }

  clError = clSetKernelArg(xinfo->kernels[0], 2, sizeof(cl_mem), &kdata->r_mem);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 2 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }
  
  clError = clSetKernelArg(xinfo->kernels[0], 3, sizeof(cl_mem), &kdata->v_mem);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 3 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }
  
  clError = clSetKernelArg(xinfo->kernels[0], 4, sizeof(cl_mem), &kdata->Av_mem);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 4 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }

  clError = clSetKernelArg(xinfo->kernels[0], 5, sizeof(cl_uint), &n);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 5 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }

  clError = clSetKernelArg(xinfo->kernels[0], 6, sizeof(cl_uint), &N);
  if(clError) 
  {
    KRYLOG_DEV(KRY_FAIL,
        "%s: error setting kernel argument 6 - clError %d",
        __func__, clError);
    return KRY_SET_KERNEL_ARG_ERROR;
  }

  return KRY_SUCCESS;
}

int kryKexec_mulSmDv(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A)
{
  size_t global_sz, local_sz;
  int err = kryKshape_mulSmDv(ginfo, A, &global_sz, &local_sz);
  if(err) return err;
  clError = clEnqueueNDRangeKernel(
      ginfo->q,
      xinfo->kernels[0],
      1,
      0,
      &global_sz,
      &local_sz,
      0, NULL, &xinfo->kcomplete);
  if(clError) return KRY_EXEC_KERNEL_ERROR;

  return KRY_SUCCESS;
}

int kryKshape_mulSmDv_simple(kryGPUInfo *ginfo, krySparseMatrix *A, size_t *gsz, 
    size_t *lsz)
{
  size_t d0sz = ginfo->max_work_item_sizes[0];
  *lsz = A->n >= d0sz ? d0sz : A->n;
  *gsz = A->n;
  return KRY_SUCCESS;
}

int kryKshape_mulSmDv_even(kryGPUInfo *ginfo, krySparseMatrix *A, size_t *gsz, 
    size_t *lsz)
{
  size_t d0sz = ginfo->max_work_item_sizes[0];


  
  return KRY_SUCCESS;
}
int kryKshape_mulSmDv(kryGPUInfo *ginfo, krySparseMatrix *A, size_t *gsz, 
    size_t *lsz)
{
  return kryKshape_mulSmDv_simple(ginfo, A, gsz, lsz);
}

int kryKresult_mulSmDv(kryGPUInfo *ginfo, kryKdata_mulSmDv *kdata, 
    krySparseMatrix *A, double **v)
{
  *v = (double*)malloc(sizeof(double)*A->n);
  clError = clEnqueueReadBuffer(
      ginfo->q,
      kdata->Av_mem,
      CL_TRUE,
      0,
      sizeof(double)*A->n,
      *v,
      0,NULL,NULL);
  if(clError) return KRY_KERNEL_READBACK_ERROR;

  return KRY_SUCCESS;
}

int kryMulSmDv (kryGPUInfo *ginfo, krySparseMatrix *A, double *v, double **Av, 
  kryExecInfo *_xinfo)
{
  kryExecInfo xinfo;
  xinfo.ginfo = ginfo;
  kryKdata_mulSmDv kdata; 

  int err = kryKalloc_mulSmDv(ginfo, &kdata, A, v);
  if(err) return err;

  err = kryKload_mulSmDv(&xinfo, &kdata, A->N, A->n);
  if(err) return err;

  err = kryKexec_mulSmDv(ginfo, &xinfo, A);
  if(err) return err;

  err = kryKresult_mulSmDv(ginfo, &kdata, A, Av);
  if(err) return err; 

  if(_xinfo) *_xinfo = xinfo;

  return KRY_SUCCESS;
}
