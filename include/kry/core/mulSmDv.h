#ifndef KRYCL_MULSMDV_H
#define KRYCL_MULSMDV_H

#include "krycl.h"

typedef struct kryKdata_mulSmDv
{
  cl_mem A_mem, c_mem, r_mem, v_mem, Av_mem;
  size_t n, N;
} 
kryKdata_mulSmDv;

int kryKalloc_mulSmDv (kryGPUInfo *ginfo, kryKdata_mulSmDv *kdata, krySparseMatrix *A, double *v);
int kryKload_mulSmDv (kryExecInfo *xinfo, kryKdata_mulSmDv *kdata, cl_uint N, cl_uint n);
int kryKexec_mulSmDv (kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A);
int kryKshape_mulSmDv (kryGPUInfo *ginfo, krySparseMatrix *A, size_t *gsz, size_t *lsz);
int kryKresult_mulSpDv (kryGPUInfo *ginfo, kryKdata_mulSmDv *kdata, krySparseMatrix *A, double **v);
int kryMulSmDv (kryGPUInfo *ginfo, krySparseMatrix *A, double *v, double **Av, kryExecInfo *xinfo);

#endif
