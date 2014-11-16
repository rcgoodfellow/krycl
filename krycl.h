#ifndef KRYCL_H
#define KRYCL_H

//#define _XOPEN_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _BSD_SOURCE
#define _XOPEN_SOURCE

#include </usr/local/include/CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define KRYCL_VERSION_MAJOR 0
#define KRYCL_VERSION_MINOR 1

#define KRY_SUCCESS 0
#define KRY_CL_PLATFORM_CREATE_ERROR -1000
#define KRY_NO_AVAILABLE_PLATFORMS -1001
#define KRY_CL_GPU_CREATE_ERROR -1002
#define KRY_NO_AVAILABLE_GPUS -1003
#define KRY_CL_CONTEXT_CREATE_ERROR -1004
#define KRY_CL_COMMAND_Q_CREATE_ERROR -1005
#define KRY_KALLOC_ERROR -1006
#define KRY_CL_SOURCE_NOT_FOUND -1007
#define KRY_INCOMPLETE_SOURCE_READ -1008
#define KRY_CREATE_PROGRAM_ERROR -1009
#define KRY_BUILD_PROGRAM_ERROR -1010
#define KRY_BUILD_LOG_ACCESS_ERROR -1011
#define KRY_CREATE_CL_KERNEL_ERROR -1012
#define KRY_SET_KERNEL_ARG_ERROR -1013
#define KRY_CL_DEVINFO_QUERY_ERROR -1014
#define KRY_EXEC_KERNEL_ERROR -1015
#define KRY_KERNEL_READBACK_ERROR -1016

extern cl_int clError;
extern FILE* kryLog;

typedef struct kryGPUInfo
{
  cl_platform_id pid;
  cl_device_id did;
  cl_context ctx;
  cl_command_queue q;
  size_t max_work_item_sizes[3];
  cl_uint max_compute_units;
  cl_program kry_core;
} 
kryGPUInfo;

typedef struct krySparseMatrix
{
  double *v;
  cl_uint *c, *r;
  cl_uint N, n;
} 
krySparseMatrix;

typedef struct kryExecInfo
{
  cl_kernel *kernels;
  kryGPUInfo *ginfo;
} 
kryExecInfo;

int kryInit();
int kryGetAGPU(kryGPUInfo *ginfo); 
int kryArnoldi(kryGPUInfo *ginfo, kryExecInfo *xinfo, krySparseMatrix *A, double *b, double *x0, double *x);
int kryCLCLog(kryGPUInfo *ginfo, cl_program prog);
int kryReadProgramSource (const char* fn, char **src, size_t *sz);
int kryLoadCore (kryGPUInfo *ginfo);

void kryPrintSparseMatrix(FILE *f, const krySparseMatrix A);
void kryPrintVecD(FILE *f, const double *v, size_t sz);
void kryPrintVecU(FILE *f, const cl_uint *v, size_t sz);

#define KRY_FAIL "fail"
#define KRY_WARN "warn"
#define KRY_INFO "info"

void kryLogTime();

#define KRYLOG_DEV(__SEV__, __FMT__, ...)                             \
  kryLogTime();                                                       \
  fprintf(kryLog, "[%s] %s:%d ", __SEV__, __FILE__, __LINE__);        \
  fprintf(kryLog, __FMT__, __VA_ARGS__);                              \
  fprintf(kryLog, "%s", "\n"); 

#define KRYLOG(__SEV__, __FMT__, ...)                                 \
  kryLogTime();                                                       \
  fprintf(kryLog, "[%s] ", __SEV__);                                  \
  fprintf(kryLog, __FMT__, __VA_ARGS__);                              \
  fprintf(kryLog, "%s", "\n"); 


#endif
