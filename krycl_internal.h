#ifndef KRYCL_INTERNAL_H
#define KRYCL_INTERNAL_H

typedef struct _arnoldiMem
{
  cl_mem Av_mem, Ac_mem, Ar_mem, b_mem, x0_mem, x_mem, r0_mem;
} _arnoldiMem;

int _arnoldiAllocateMem(kryGPUInfo *ginfo, _arnoldiMem *amem, 
    krySparseMatrix *A, double *b, double *x0);

int _arnoldiLoadCLProgram(kryGPUInfo *ginfo, kryExecInfo *xinfo);

int _readProgramSource(const char* fn, char **src, size_t *sz);

int _arnoldiLoadKernels(kryExecInfo *xinfo, _arnoldiMem *amem, cl_uint N,
    cl_uint n);

int _arnoldiExecute(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A,
    _arnoldiMem *amem);

int _exec_mul_sp_dv(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A);

int _getShape_mul_sp_dv(kryGPUInfo *ginfo, krySparseMatrix *A, 
    size_t *gsz, size_t *lsz);

int _getResult_mul_sp_dv(kryGPUInfo *ginfo, _arnoldiMem *amem, 
    krySparseMatrix *A, double **Av);

#endif
