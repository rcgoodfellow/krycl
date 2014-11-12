#ifndef KRYCL_INTERNAL_H
#define KRYCL_INTERNAL_H

typedef struct _arnoldiMem
{
  cl_mem Av_mem, Ac_mem, Ar_mem, b_mem, x0_mem, x_mem;
} _arnoldiMem;

int _arnoldiAllocateMem(kryGPUInfo *ginfo, _arnoldiMem *amem, 
    krySparseMatrix *A, double *b, double *x0);

int _arnoldiLoadCLProgram(kryGPUInfo *ginfo, cl_program *prog);

int _readProgramSource(const char* fn, char **src, size_t *sz);

#endif
